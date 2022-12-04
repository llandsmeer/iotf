import numpy as np
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['GraphcoreStreamingRunner']

STEP40_TEMPLATE = '''
@tf.function(jit_compile=True)
def step40(nms, state {args}):
    def inner_loop_body(state):
        return timestep(state=state {kwargs})['state_next']
    def outer_loop_body(state):
        state = ipu.loops.repeat(40, inner_loop_body, state)
        return state, outfeed_queue.enqueue(state[0,:])
    state = ipu.loops.repeat(nms, outer_loop_body, state)
    return state
'''

class GraphcoreStreamingRunner(BaseRunner):
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
        self.outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
            ipu.ipu_outfeed_queue.IPUOutfeedMode.ALL,
            buffer_depth=200
        )
        env = dict(
            tf=tf,
            ipu=ipu,
            timestep=timestep,
            outfeed_queue=self.outfeed_queue
        )
        exec(src, env)
        return env['step40']

    def setup(self, *, ngj, ncells, argconfig):
        from tensorflow.python import ipu
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.optimizations.math.fast = True
        config.optimizations.enable_gather_simplifier = True
        config.io_tiles.num_io_tiles = 128
        config.io_tiles.place_ops_on_io_tiles = True
        config.ipu_model.compile_ipu_code = True
        config.configure_ipu_system()
        self.strategy = ipu.ipu_strategy.IPUStrategy()
        with self.strategy.scope():
            self.step40 = self.compile_step40(ngj=ngj, ncells=ncells, argconfig=argconfig)
            self.args = list(argconfig.keys())

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        from tensorflow.python import ipu
        with self.strategy.scope():
            state_next = self.strategy.run(self.step40, (nms, state), kwargs)
            if probe:
                trace = self.outfeed_queue.dequeue(wait_for_completion=True).numpy()
        if probe:
            return state_next, np.vstack(((state[0,:],), trace))
        else:
            return state_next


    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        from tensorflow.python import ipu
        with self.strategy.scope():
            state_next = self.strategy.run(self.step40, (nms, state, gj_src, gj_tgt, g_gj), kwargs)
            if probe:
                trace = self.outfeed_queue.dequeue(wait_for_completion=True).numpy()
        if probe:
            return state_next, np.vstack(((state[0,:],), trace))
        else:
            return state_next

