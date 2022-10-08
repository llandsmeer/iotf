import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('/home/llandsmeer/repos/llandsmeer/iotf')

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
import time

import builder3d

config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()
config.optimizations.math.fast = True

def get_slowdown(nneurons, gj=True, repeat=1):
    if gj:
        gj_src, gj_tgt = builder3d.sample_connections_3d(
                nneurons, rmax=4)
    else:
        gj_src, gj_tgt = [], []

    state = builder3d.make_initial_neuron_state(nneurons, dtype=tf.float32)

    tf_function = builder3d.make_function(
        ngj=len(gj_src),
        ncells=nneurons,
        argconfig=dict(
        ))

    T = .2
    measurements = []
    if gj:
        def f(state):
            return tf_function(state=state, gj_src=gj_src, gj_tgt=gj_tgt, g_gj=0.05)
    else:
        def f(state):
            return tf_function(state=state)
    if repeat != 1:
        def g(state):
            return ipu.loops.repeat(repeat, f, state)
    else:
        g = f
    for _ in range(10):
        g(state)
    for _ in range(10):
        a = time.time()
        niter = 0
        while time.time() - a <= T or niter < 2:
            state = g(state)
            niter += repeat
        measurements.append(round(40000 / niter * (time.time() - a), 3))
    return np.min(measurements)

if __name__ == '__main__':
    strategy = ipu.ipu_strategy.IPUStrategy()
    print('nogj,gj')#,nogj_repeat40,gj_repeat40')
    with strategy.scope():
        for sqrt3 in range(4, 20):
            print(sqrt3**3,
                    get_slowdown(sqrt3**3, gj=False),
                    get_slowdown(sqrt3**3, gj=True),
                    # get_slowdown(sqrt3**3, gj=False, repeat=40),
                    # get_slowdown(sqrt3**3, gj=True, repeat=40),
                    sep=',')
