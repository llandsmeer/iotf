import numpy as np
import onnxruntime as ort
import tensorflow as tf
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['TfBaseRunner']

class TfBaseRunner(BaseRunner):
    '''
    Base TF implementation.
    '''
    def is_supported(self):
        return True

    def setup(self, *, ngj, ncells, argconfig):
        
        tf_function = model.make_tf_function(ngj=ngj, ncells=ncells, argconfig=argconfig)
        if ngj == 0:
            @tf.function(jit_compile=True)
            def f(state):
                for _ in range(40):
                    state = tf_function(state)['state_next']
                return state
            self.f = f
        else:
            @tf.function(jit_compile=True)
            def f(state, gj_src, gj_tgt, g_gj):
                for _ in range(40):
                    state = tf_function(state, gj_src, gj_tgt, g_gj)['state_next']
                return state
            self.f = f

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        trace = []
        if probe:
            trace.append(state.numpy()[0, :])
        for _ in range(nms):
            state = self.f(state)
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
            state = self.f(state, gj_src, gj_tgt, g_gj)
            if probe:
                trace.append(state.numpy()[0, :])
        if probe:
            return tf.constant(state), np.array(trace)
        else:
            return tf.constant(state)
