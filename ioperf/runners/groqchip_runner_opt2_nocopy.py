import os
import tempfile
import numpy as np
import tensorflow as tf
import tf2onnx
import subprocess
import time
import json

try:
    from groq.runner import tsp
    from groq.runtime import driver as runtime
    import groq.runtime
    supported = True
except ImportError:
    supported = False

from .. import model
from .base_runner import BaseRunner

__all__ = ['GroqchipRunnerOpt2NoCopy']

class GroqchipRunnerOpt2NoCopy(BaseRunner):
    '''
    GroqChip implementation
    using on chip persistent data.
    '''

    def is_supported(self):
        return supported

    def setup(self, ngj, ncells, argconfig):
        self.dtype = tf.float32
        self.ncells = ncells
        self.onnx_path = model.make_onnx_model(ngj=ngj, ncells=ncells, argconfig=argconfig)
        self.onnxFileTo   = self.onnx_path[:-5] + "_onnxTo.onnx"
        self.onnxFileFrom = self.onnx_path[:-5] + "_onnxFrom.onnx"
        self.iop_path = tempfile.mktemp() + ".iop"
        self.aa_path = tempfile.mktemp()
        self.compiler_stats = {}
        self.compile()
        self.assemble()

        #Groq Chip with presistent data
        self.shim = groq.runtime.DriverShim()
        self.dptr = self.shim.next_available_device()
        self.dptr.open()
        self.prog = runtime.IOProgram(self.iop_path)
        self.dptr.load(self.prog[0])
        self.dptr.load(self.prog[1], unsafe_keep_entry_points=True)
        self.dptr.load(self.prog[2], unsafe_keep_entry_points=True)

        # create all needed DMA buffers
        self.inputs_in    = runtime.BufferArray(self.prog[0].entry_points[0].input, 1)
        self.outputs_in   = runtime.BufferArray(self.prog[0].entry_points[0].output, 1)

        self.inputs_comp  = runtime.BufferArray(self.prog[1].entry_points[0].input, 1)
        self.outputs_comp = runtime.BufferArray(self.prog[1].entry_points[0].output, 1)

        self.inputs_out   = runtime.BufferArray(self.prog[2].entry_points[0].input, 1)
        self.outputs_out  = runtime.BufferArray(self.prog[2].entry_points[0].output, 1)

    def assemble(self):
        status = subprocess.call(
            [
                "aa-latest",
                "--name",
                f"DataMoverIn",
                "-i",
                self.aa_path   + ".0.aa",
                "--output-iop",
                self.iop_path  + ".0.iop",
                "--no-metrics",
                "--auto-agt",
                "--auto-repeat",
                "--ifetch-slice-ordering",
                "round-robin",
            ],
            stdout=open(os.devnull, "wb"),
        )
        if status != 0:
            raise Exception("Assembler call failed")

        status = subprocess.call(
            [
                "aa-latest",
                "--name",
                f"IOTimestep",
                "-i",
                self.aa_path + ".1.aa",
                "--output-iop",
                self.iop_path  + ".1.iop",
                "--no-metrics",
                "--program-package",
                self.iop_path  + ".0.iop",
                "--auto-agt",
                "--auto-repeat",
                "--ifetch-slice-ordering",
                "round-robin",
            ],
            stdout=open(os.devnull, "wb"),
        )
        if status != 0:
            raise Exception("Assembler call failed")

        status = subprocess.call(
            [
                "aa-latest",
                "--name",
                f"DataMoverOut",
                "-i",
                self.aa_path + ".2.aa",
                "--output-iop",
                self.iop_path,
                "--no-metrics",
                "--auto-agt",
                "--auto-repeat",
                "--program-package",
                self.iop_path  + ".1.iop",
                "--ifetch-slice-ordering",
                "round-robin",
            ],
            stdout=open(os.devnull, "wb"),
        )
        if status != 0:
            raise Exception("Assembler call failed")

        print(f"[GroqAssembler] Finished {self.iop_path}")

    def compile(self):
        # [ONNX] Data MOVER constants


        # [ONNX] DATA MOVER to TSP
        onnx_model, _ = tf2onnx.convert.from_function(
            function=self.mkid('result_mover_in'),
            input_signature=(tf.TensorSpec((14,self.ncells), self.dtype, name="input_mover_in"),),
            output_path=self.onnxFileTo,
            opset=16,
        )
        # [ONNX] DATA MOVER from TSP
        onnx_model, _ = tf2onnx.convert.from_function(
            function=self.mkid('result_mover_out'),
            input_signature=(tf.TensorSpec((14,self.ncells), self.dtype, name="input_mover_out"),),
            output_path=self.onnxFileFrom,
            opset=16,
        )

        print("[GroqCompiler] Starting Groq compiler")
        status = subprocess.call(
            [
            "groq-compiler",
            f"-save-stats={self.aa_path}_compilerstats",
            "--coresident",
            "--persistent=result_mover_in,input_mover_out,state_next,state",
            "-o",
            self.aa_path,
            self.onnxFileTo,
            self.onnx_path,
            self.onnxFileFrom,
            ],
            stdout=open(os.devnull, "wb"),
        )
        if status != 0:
            raise Exception("Compiler call failed")

        print(self.aa_path)
        # with open(f"{self.aa_path}_compilerstats", "r") as f:
        #     self.compiler_stats = json.load(f)

    # Helper function
    def mkid(self,name):
        def tf_identity(x):
            return {name: x}
        return tf.function(tf_identity)

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        trace = []
        state  = np.array(state)
        if probe:
            trace.append(state[0, :].copy())

        # Move data to the chip
        self.prog[0].entry_points[0].input.tensors[0].from_host(state, self.inputs_in[0])
        self.dptr.invoke(self.inputs_in[0],self.outputs_in[0])

        # Run compute and move data back if needed.
        for _ in range(nms):
            for _ in range(40):
                self.dptr.invoke(self.inputs_comp[0],self.outputs_comp[0])
            #get data out of chip!
            self.dptr.invoke(self.inputs_out[0],self.outputs_out[0])
            self.prog[2].entry_points[0].output.tensors[0].to_host(self.outputs_out[0],state)
            if probe:
                trace.append(state[0, :].copy())

        # Finished
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)

    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        trace = []
        state  = np.array(state)
        gj_src = np.array(gj_src)
        gj_tgt = np.array(gj_tgt)
        g_gj   = np.array(g_gj,dtype=np.float32).reshape(1,1)

        if probe:
            trace.append(state[0, :].copy())

        # Move data to the chip
        self.prog[0].entry_points[0].input.tensors[0].from_host(state, self.inputs_in[0])
        self.dptr.invoke(self.inputs_in[0],self.outputs_in[0])

        # FIXEN dit moet in de move data IOP!
        self.prog[1].entry_points[0].input.tensors[0].from_host(gj_src, self.inputs_comp[0])
        self.prog[1].entry_points[0].input.tensors[1].from_host(gj_tgt, self.inputs_comp[0])
        self.prog[1].entry_points[0].input.tensors[2].from_host(g_gj,   self.inputs_comp[0])

        # Run compute and move data back if needed.
        for _ in range(nms):
            for _ in range(40):
                self.dptr.invoke(self.inputs_comp[0],self.outputs_comp[0])
            #move data back to the cpu
            self.dptr.invoke(self.inputs_out[0],self.outputs_out[0])
            self.prog[2].entry_points[0].output.tensors[0].to_host(self.outputs_out[0],state)
            if probe:
                trace.append(state[0, :].copy())

        # Finished
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)