import numpy as np
import onnxruntime as ort
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['TfBaseRunner']

STEP40_TEMPLATE = '''
@tf.function(jit_compile=True)
def f(state {args}):
    for _ in range(40):
        state = tf_function(state {kwargs})['state_next']
    return state
'''

class TfBaseRunner(BaseRunner):
    '''
    Base TF implementation.
    '''
    def is_supported(self):
        return True

    def setup(self, *, ngj, ncells, argconfig):
        tf_function = model.make_tf_function(ngj=ngj, ncells=ncells, argconfig=argconfig)
        args = list(argconfig.keys())
        if ngj != 0:
            args = ['gj_src', 'gj_tgt', 'g_gj'] + args
        prefix = ', ' if args else ''
        src = STEP40_TEMPLATE.format(
                args=prefix + ', '.join(args),
                kwargs=prefix + ', '.join(f'{k}={k}' for k in args)
                )
        env = dict(
            tf=tf,
            tf_function=tf_function
        )
        exec(src, env)
        self.f = env['f']

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        trace = []
        if probe:
            trace.append(state.numpy()[0, :])
        for _ in range(nms):
            state = self.f(state, **kwargs)
            if probe:
                trace.append(state.numpy()[0, :])
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)

    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        trace = []
        if probe:
            trace.append(state.numpy()[0, :])
        for _ in range(nms):
            state = self.f(state, gj_src, gj_tgt, g_gj, **kwargs)
            if probe:
                trace.append(state.numpy()[0, :])
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)
