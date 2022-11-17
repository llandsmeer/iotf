import numpy as np
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['GraphcoreRunner']

class GraphcoreRunner(BaseRunner):
    def setup(self, *args, **kwargs):
        from tensorflow.python import ipu
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 2
        config.configure_ipu_system()
        config.optimizations.math.fast = True
        self.tf_function = model.make_tf_function(*args, **kwargs)

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        from tensorflow.python import ipu
        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            trace = []
            timestep = self.tf_function
            def loop_body(state):
                return timestep(state=state, **kwargs)
            for _ in range(nms):
                state = ipu.loops.repeat(40, loop_body, state)
                if probe:
                    trace.append(state[:,0].numpy())
        if probe:
            return state, np.array(trace)
        else:
            return state


    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        from tensorflow.python import ipu
        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            trace = []
            timestep = self.tf_function
            def loop_body(state):
                return timestep(state=state, gj_src=gj_src, gj_tgt=gj_tgt, g_gj=g_gj, **kwargs)
            for _ in range(nms):
                state = ipu.loops.repeat(40, loop_body, state)
                if probe:
                    trace.append(state[:,0].numpy())
        if probe:
            return state, np.array(trace)
        else:
            return state

