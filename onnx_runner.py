import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sys.path.append('/home/llandsmeer/repos/llandsmeer/iotf')

import onnx
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import time
import builder3d

# the options.
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

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

    onnx.checker.check_model(onnx_model)
    #print("==== ONNX graph starts ====")
    #print(onnx.helper.printable_graph(onnx_model.graph))
    #print("==== ONNX graph ends ====")

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

    print( ort.get_device()  ) 
    ort_sess = ort.InferenceSession("/tmp/io.onnx", sess_options, providers=EP_list)
    io_binding = ort_sess.io_binding()
    
    io_binding.bind_cpu_input('state',  state.numpy())
    io_binding.bind_cpu_input('gj_src', gj_src.numpy())
    io_binding.bind_cpu_input('gj_tgt', gj_tgt.numpy())
    io_binding.bind_cpu_input('g_gj',   np.array(0.05, dtype='float32'))

    #change device type here to cuda if cuda
    io_binding.bind_output('Identity:0',device_type="cpu")

    a = time.time()
    niter = 0
    while time.time() - a <= 1:
        ort_sess.run_with_iobinding(io_binding)
        #outputs = ort_sess.run(None, {
        #    'state': state.numpy(),
        #    'gj_src': gj_src.numpy(),
        #    'gj_tgt': gj_tgt.numpy(),
        #    'g_gj': np.array(0.05, dtype='float32')
        #    })        
        niter += 1
        if(not(niter%40)):
            Y = io_binding.copy_outputs_to_cpu()

    return 40000 / niter

if __name__ == '__main__':
    results = []
    print("starting:")
    for sqrt3 in 4,5,6,7,8,9,10:
        result = get_slowdown(sqrt3**3)
        print(sqrt3**3, result)
        results.append(result)
    print(results)
