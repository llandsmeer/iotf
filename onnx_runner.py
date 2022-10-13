import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sys.path.append('/home/llandsmeer/repos/llandsmeer/iotf')

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import time

import builder3d

print('Backend', ort.get_device())

def get_slowdown(nneurons, par=False, gj=True):
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

    onnx_model, _ = tf2onnx.convert.from_function(
            function=tf_function, 
            input_signature=tf_function.argspec,
            output_path='/tmp/io.onnx',
            opset=16,
            )

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.inter_op_num_threads = 0
    if par:
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    ort_sess = ort.InferenceSession('/tmp/io.onnx', sess_options=opts)
    a = time.time() # WRONG PLACE
    T = .1
    state = ort.OrtValue.ortvalue_from_numpy(state.numpy())
    args = { 'state': state.numpy() }
    if gj:
        src = ort.OrtValue.ortvalue_from_numpy(gj_src.numpy())
        tgt = ort.OrtValue.ortvalue_from_numpy(gj_tgt.numpy())
        gj = ort.OrtValue.ortvalue_from_numpy(np.array(0.05, dtype='float32'))
        args['gj_src'] = src
        args['gj_tgt'] = tgt
        args['g_gj'] = gj # SHOULD BE 0.05!!!!!!!!
    measurements = []
    for _ in range(10):
        niter = 0
        while time.time() - a <= T:
            outputs = ort_sess.run(None, args)
            niter += 1
        measurements.append(round(40000 / niter * T, 3)) # Dont multiply by T but by elapsed time
    return np.min(measurements), np.max(measurements), np.mean(measurements), np.std(measurements)

if __name__ == '__main__':
    for sqrt3 in range(4, 20):
        print(sqrt3**3,
                get_slowdown(sqrt3**3, par=False, gj=False),
                get_slowdown(sqrt3**3, par=False, gj=True),
                get_slowdown(sqrt3**3, par=True, gj=True),
                sep=',')
