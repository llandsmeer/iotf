import os

import tempfile
import numpy as np
import tensorflow as tf
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

__all__ = ['GroqchipRunnerOpt1']

class GroqchipRunnerOpt1(BaseRunner):
    '''
    GroqChip implementation
    compiled 40 timesteps in 1 iop
    '''

    def is_supported(self):
        return supported

    def setup(self, ngj, ncells, argconfig):
        self.onnx_path = model.make_onnx_model(ngj=ngj, ncells=ncells, argconfig=argconfig, combine40=True)
        self.iop_path = tempfile.mktemp() + ".iop"
        self.aa_path = tempfile.mktemp()
        self.compiler_stats = {}
        self.compile()
        self.assemble()
        self.program = tsp.create_tsp_runner(self.iop_path)

    def assemble(self):
        print(f"[groqAssembler] Starting Groq compiler {self.iop_path}")
        status = subprocess.call(
            [
                "aa-latest",
                "--name",
                f"IOTimestep",
                "--large-program",
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
        print(f"[GroqCompiler] Starting Groq compiler input onnx {self.onnx_path}")
        status = subprocess.call(
            [
            "groq-compiler",
            f"-save-stats={self.aa_path}_compilerstats",
            "--large-program",
            # "--groqview",
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
        state = state.numpy()

        if probe:
            trace.append(state[0, :])

        kwargs = {k.lower(): v for k, v in kwargs.items()}

        for _ in range(nms):
            # for _ in range(40):
            state = self.program(state=state, **kwargs)["state_next"]
            if probe:
                trace.append(state[0, :])

        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)

    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        trace = []
        state = state.numpy()
        gj_src = gj_src.numpy()
        gj_tgt = gj_tgt.numpy()
        g_gj = np.array(g_gj,dtype=np.float32).reshape(1,1)
        kwargs = {k.lower(): v for k, v in kwargs.items()}

        if probe:
            trace.append(state[0, :])

        for _ in range(nms):
            # for _ in range(40):
            state = self.program(state=state,gj_src=gj_src,gj_tgt=gj_tgt,g_gj=g_gj,**kwargs)["state_next"]
            if probe:
                trace.append(state[0, :])

        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)
