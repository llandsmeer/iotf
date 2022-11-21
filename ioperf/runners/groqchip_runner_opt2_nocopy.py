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

        #Groq Chip with presistent data
        shim = groq.runtime.DriverShim()
        dptr = shim.next_available_device()
        dptr.open()
        prog = runtime.IOProgram(self.iop_path)
        dptr.load(prog[0])
        dptr.load(prog[1], unsafe_keep_entry_points=True)
        dptr.load(prog[2], unsafe_keep_entry_points=True)
        
        # create all needed DMA buffers
        inputs_in   = runtime.BufferArray(prog[0].entry_points[0].input, 1)
        outputs_in = runtime.BufferArray(prog[0].entry_points[0].output, 1)
        inputs_comp = runtime.BufferArray(prog[1].entry_points[2].input, 1)
        outputs_comp = runtime.BufferArray(prog[1].entry_points[2].output, 1)
        inputs_out   = runtime.BufferArray(prog[2].entry_points[0].input, 1)
        outputs_out = runtime.BufferArray(prog[2].entry_points[0].output, 1)
     
        # Move data to the chip
        prog[0].entry_points[0].input.tensors[0].from_host(state, inputs_in[0])
        dptr.invoke(inputs_in[0],outputs_in[0])
        
        # Run compute and move data back if needed.
        for _ in range(nms):
            for _ in range(40):
                dptr.invoke(inputs_comp[0],outputs_comp[0])
            dptr.invoke(inputs_out[0],outputs_out[0])
            prog[2].entry_points[0].output.tensors[0].to_host(outputs_out[0],state)
            if probe:
                trace.append(state[0, :])

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
        g_gj = np.array(g_gj,dtype=np.float32).reshape(1,1)        

        #Groq Chip with presistent data
        shim = groq.runtime.DriverShim()
        dptr = shim.next_available_device()
        dptr.open()
        prog = runtime.IOProgram(self.iop_path)
        dptr.load(prog[0])
        dptr.load(prog[1], unsafe_keep_entry_points=True)
        dptr.load(prog[2], unsafe_keep_entry_points=True)
        
        # create all needed DMA buffers
        inputs_in   = runtime.BufferArray(prog[0].entry_points[0].input, 1)
        outputs_in = runtime.BufferArray(prog[0].entry_points[0].output, 1)

        inputs_comp_in = runtime.BufferArray(prog[1].entry_points[1].input, 1)
        outputs_comp_in = runtime.BufferArray(prog[1].entry_points[1].output, 1)
        
        inputs_comp = runtime.BufferArray(prog[1].entry_points[2].input, 1)
        outputs_comp = runtime.BufferArray(prog[1].entry_points[2].output, 1)
        
        inputs_out   = runtime.BufferArray(prog[2].entry_points[0].input, 1)
        outputs_out = runtime.BufferArray(prog[2].entry_points[0].output, 1)
     
        # Move data to the chip
        prog[0].entry_points[0].input.tensors[0].from_host(state, inputs_in[0])
        dptr.invoke(inputs_in[0],outputs_in[0])
        prog[1].entry_points[1].input.tensors[0].from_host(gj_src, inputs_comp_in[0])
        prog[1].entry_points[1].input.tensors[1].from_host(gj_tgt, inputs_comp_in[0])
        prog[1].entry_points[1].input.tensors[2].from_host(g_gj, inputs_comp_in[0])
        dptr.invoke(inputs_comp_in[0],outputs_comp_in[0])

        
        # Run compute and move data back if needed.
        for _ in range(nms):
            for _ in range(40):
                dptr.invoke(inputs_comp[0],outputs_comp[0])
            dptr.invoke(inputs_out[0],outputs_out[0])
            prog[2].entry_points[0].output.tensors[0].to_host(outputs_out[0],state)
            if probe:
                trace.append(state[0, :])

        # Finished
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)
