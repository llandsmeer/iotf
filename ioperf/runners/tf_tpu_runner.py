import numpy as np
import onnxruntime as ort
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['TfTpuRunner']

STEP40_TEMPLATE = '''
@tf.function(jit_compile=True)
def f(state {args}):
    for _ in range(40):
        state = tf_function(state {kwargs})['state_next']
    return state
'''

class TfTpuRunner(BaseRunner):
    '''
    TPU TF implementation.
    '''
    def __init__(self):
        # setup tpu google
        self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(self.resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(self.resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))

    def is_supported(self):
        return True

    def setup(self, *, ngj, ncells, argconfig):
        with tf.device('/TPU:0'):
            # rest of the setup
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
        with tf.device('/TPU:0'):
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
        with tf.device('/TPU:0'):
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
