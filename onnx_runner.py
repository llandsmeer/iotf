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

def get_slowdown(nneurons):
    gj_src, gj_tgt = builder3d.sample_connections_3d(
            nneurons, rmax=4)
    state = builder3d.make_initial_neuron_state(nneurons, dtype=tf.float32, V_axon=None, V_dend=None, V_soma=None)

    tf_function = builder3d.make_function(
        ngj=len(gj_src),
        ncells=nneurons,
        argconfig=dict(
        #    g_CaL = 'VARY'
        ))

    onnx_model, _ = tf2onnx.convert.from_function(
            function=tf_function, 
            input_signature=tf_function.argspec,
            output_path='/tmp/io.onnx',
            opset=16,
            )

    ort_sess = ort.InferenceSession('/tmp/io.onnx')
    a = time.time()
    niter = 0
    while time.time() - a <= 1:
        outputs = ort_sess.run(None, {
            'state': state.numpy(),
            'gj_src': gj_src.numpy(),
            'gj_tgt': gj_tgt.numpy(),
            'g_gj': np.array(0.05, dtype='float32')
            })
        niter += 1
    return 40000 / niter

if __name__ == '__main__':
    for sqrt3 in 4,5,6,7,8,9,10:
        print(sqrt3**3, get_slowdown(sqrt3**3))
