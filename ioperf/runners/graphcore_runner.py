import numpy as np
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['GraphcoreRunner']

class GraphcoreRunner(BaseRunner):
    def is_supported(self):
        try:
            from tensorflow.python import ipu
        except ImportError as ex:
            print(repr(ex))
            return False
        return True

    def setup(self, *, ngj, ncells, argconfig):
        from tensorflow.python import ipu
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 2
        config.configure_ipu_system()
        config.optimizations.math.fast = True
        self.strategy = ipu.ipu_strategy.IPUStrategy()
        with self.strategy.scope():
            timestep = model.make_tf_function(ngj=ngj, ncells=ncells, argconfig=argconfig)
            if ngj == 0:
                @tf.function(jit_compile=True)
                def step40(state):
                    def loop_body(state):
                        return timestep(state=state)['state_next']
                    return ipu.loops.repeat(40, loop_body, state)
                self.step40 = step40
            else:
                @tf.function(jit_compile=True)
                def step40(state, gj_src, gj_tgt, g_gj):
                    def loop_body(state):
                        return timestep(state=state, gj_src=gj_src, gj_tgt=gj_tgt, g_gj=g_gj)['state_next']
                    return ipu.loops.repeat(40, loop_body, state)
                self.step40 = step40

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        from tensorflow.python import ipu
        with self.strategy.scope():
            trace = [state[0,:].numpy()]
            for _ in range(nms):
                state = self.strategy.run(self.step40, (state,))
                if probe:
                    trace.append(state[0,:].numpy())
        if probe:
            return state, np.array(trace)
        else:
            return state


    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        from tensorflow.python import ipu
        with self.strategy.scope():
            trace = []
            trace.append(state[0,:].numpy())
            for _ in range(nms):
                state = self.strategy.run(self.step40, (state, gj_src, gj_tgt, gj_gj))
                if probe:
                    trace.append(state[0,:].numpy())
        if probe:
            return state, np.array(trace)
        else:
            return state

