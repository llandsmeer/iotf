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

__all__ = ['GroqchipRunner_opt2_nocopy']

class GroqchipRunner(BaseRunner):
    '''
    GroqChip implementation
    using on chip persistent data.
    '''
    def setup(self, ngj, ncells, argconfig):
        self.onnx_path = model.make_onnx_model(ngj=ngj, ncells=ncells, argconfig=argconfig)
        self.iop_path = tempfile.mktemp() + ".iop"
        self.aa_path = tempfile.mktemp()
        self.compiler_stats = {}
        self.compile()
        self.assemble()

    def assemble(self):
        raise NotImplementedError()

    def compile(self):
        raise NotImplementedError()

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        raise NotImplementedError()

    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        raise NotImplementedError()