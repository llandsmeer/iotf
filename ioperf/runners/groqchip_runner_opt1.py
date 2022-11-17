import os

import tempfile
import numpy as np
import tensorflow as tf
import subprocess
import time
import json

from groq.runner import tsp
from groq.runtime import driver as runtime
import groq.runtime

from .. import model
from .base_runner import BaseRunner

__all__ = ['GroqchipRunner_opt1']

class GroqchipRunner(BaseRunner):
    '''
    GroqChip implementation
    Running a with better dma buffer control this only works for NO gap junctions, dma buffer swapping only works for same buffer size
    '''

    def is_supported(self):
        return os.system('which groq-compiler') == 0

    def setup(self, ngj, ncells, argconfig):
        self.onnx_path = model.make_onnx_model(ngj=ngj, ncells=ncells, argconfig=argconfig)
        self.iop_path = tempfile.mktemp() + ".iop"
        self.aa_path = tempfile.mktemp()
        self.compiler_stats = {}
        self.compile()
        self.assemble()

    def assemble(self):
        print(f"[groqAssembler] Starting Groq compiler {self.iop_path}")
        status = subprocess.call(
            [
                "aa-latest",
                "--name",
                f"IOTimestep",
                "-i",
                self.aa_path + ".aa",
                "--output-iop",
                self.iop_path,
            ],
            stdout=open(os.devnull, "wb"),
        )
        if status != 0:
            raise Exception("Assembler call failed")

    def compile(self):
        print("[GroqCompiler] Starting Groq compiler")
        status = subprocess.call(
            [
            "groq-compiler",
            f"-save-stats={self.aa_path}_compilerstats",
            "-o",
            self.aa_path,
            self.onnx_path,
            ],
            stdout=open(os.devnull, "wb"),
        )
        if status != 0:
            raise Exception("Compiler call failed")

        with open(f"{self.aa_path}_compilerstats", "r") as f:
            self.compiler_stats = json.load(f)

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        trace = []
        state = np.array(state)

        shim = groq.runtime.DriverShim()
        dptr = shim.next_available_device()
        dptr.open()
        prog = runtime.IOProgram(self.iop_path)
        dptr.load(prog[0]) 

        inputs = runtime.BufferArray(prog[0].entry_points[0].input, 1)
        outputs = runtime.BufferArray(prog[0].entry_points[0].output, 1)

        inputs_iodesc.tensors[0].from_host(state, inputs[0])
   
        for _ in range(nms):
            for _ in range(40/2*nms):
                dptr.invoke(inputs[0],outputs[0])
                dptr.invoke(outputs[0],inputs[0])
            if probe:
                outputs_iodesc.tensors[0].to_host(inputs[0],state)
                trace.append(state[0, :])
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)
        
        dptr.close()
        print(data)

    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        print("[ERROR] not an option as DMA buffers are not off the same size")
        raise NotImplementedError()