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

# the options.
EP_list = ['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider']

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

    sess_options = ort.SessionOptions()
    sess_options.use_deterministic_compute = True
    
    # GRAPH optimalizations
    #   GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
    #   GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
    #   GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
    #   GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED 
    # sess_options.optimized_model_filepath = "/tmp/io_optimized_model.onnx"  #safe the optimilized graph here!

    # ENABLE PROFILING
    # sess_options.enable_profiling = True
    
    # ENABLE MULTI TREAD / NODE   (intra == openMP inside a node, INTRA == multiNODE)
    # opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # ORT_PARALLEL
    # opts.inter_op_num_threads = 0
    # opts.intra_op_num_threads = 0  #Inter op num threads (used only when parallel execution is enabled) is not affected by OpenMP settings and should always be set using the ORT APIs.

  
    ort_sess = ort.InferenceSession("/tmp/io.onnx", sess_options)

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
