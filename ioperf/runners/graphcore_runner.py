import numpy as np
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['GraphcoreRunner']

STEP40_TEMPLATE = '''
@tf.function(jit_compile=True)
def step40(state {args}):
    def loop_body(state):
        return timestep(state=state {kwargs})['state_next']
    return ipu.loops.repeat(40, loop_body, state)
'''

class GraphcoreRunner(BaseRunner):
    def is_supported(self):
        try:
            from tensorflow.python import ipu
        except ImportError as ex:
            print(repr(ex))
            return False
        return True

    def compile_step40(self, ngj, ncells, argconfig):
        from tensorflow.python import ipu
        args = list(argconfig.keys())
        if ngj != 0:
            args = ['gj_src', 'gj_tgt', 'g_gj'] + args
        timestep = model.make_tf_function(ngj=ngj, ncells=ncells, argconfig=argconfig)
        prefix = ', ' if args else ''
        src = STEP40_TEMPLATE.format(
                args=prefix + ', '.join(args),
                kwargs=prefix + ', '.join(f'{k}={k}' for k in args)
                )
        env = dict(
            tf=tf,
            ipu=ipu,
            timestep=timestep
        )
        exec(src, env)
        return env['step40']

    def setup(self, *, ngj, ncells, argconfig):
        from tensorflow.python import ipu
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 2
        config.optimizations.math.fast = True
        config.optimizations.enable_gather_simplifier = True
        config.configure_ipu_system()
        self.strategy = ipu.ipu_strategy.IPUStrategy()
        with self.strategy.scope():
            self.step40 = self.compile_step40(ngj=ngj, ncells=ncells, argconfig=argconfig)
            self.args = list(argconfig.keys())

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        from tensorflow.python import ipu
        with self.strategy.scope():
            extra = tuple(kwargs[k] for k in self.args)
            trace = [state[0,:].numpy()]
            for _ in range(nms):
                state = self.strategy.run(self.step40, (state, *extra))
                if probe:
                    trace.append(state[0,:].numpy())
        if probe:
            return state, np.array(trace)
        else:
            return state


    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        from tensorflow.python import ipu
        with self.strategy.scope():
            extra = tuple(kwargs[k] for k in self.args)
            trace = []
            trace.append(state[0,:].numpy())
            for _ in range(nms):
                args = (state, gj_src, gj_tgt, g_gj, *extra)
                state = self.strategy.run(self.step40, args)
                if probe:
                    trace.append(state[0,:].numpy())
        if probe:
            return state, np.array(trace)
        else:
            return state

