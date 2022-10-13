import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sys.path.append('/home/llandsmeer/repos/llandsmeer/iotf')

import time
import tempfile

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

import builder3d


ONNX_MODEL_PATH = tempfile.mktemp()

def run_experiment(
        nneurons,
        par=False,
        gj=True,
        do_experiment=True,
        n_measurement_repititions=10,
        g_gj=0.005,
        experiment_seconds = 5,
        randomize_cell_params=(
            ('g_CaL',   1.1,  -.6),
            ('g_int',   0.13, -0.02),
            ('g_h',     0.12,  1),
            ('g_K_Ca', 35.1, 10),
            ('g_ld',    0.01532, -0.003),
            ('g_la',    0.016, -0.003),
            ('g_ls',    0.016, -0.003),
            ),
        runner='ONNX_CPU'
        ):

    assert runner in {'ONNX_CPU', 'ONNX_CUDA'}

    # THE INTIIAL VALUE AND CONFIGURATION
    np.random.seed(0)
    if gj:
        gj_src, gj_tgt = builder3d.sample_connections_3d(
                nneurons, rmax=4)
    else:
        gj_src, gj_tgt = [], []
    state0 = builder3d.make_initial_neuron_state(nneurons, dtype=tf.float32)
    cell_config = {}
    for var, zero, delta in randomize_cell_params:
        cell_config[var] = (zero + delta * np.random.random(nneurons)).astype('float32')

    # GET A TF FUNCTION OF THE MODEL READY FOR COMPILATION
    argconfig = {}
    for var, _, _ in randomize_cell_params:
        argconfig[var] = 'VARY'
    tf_function = builder3d.make_function(
        ngj=len(gj_src),
        ncells=nneurons,
        argconfig=argconfig
        )

    # CONVERT TO ONNX
    tf2onnx.convert.from_function(
            function=tf_function, 
            input_signature=tf_function.argspec,
            output_path=ONNX_MODEL_PATH,
            opset=16,
            )

    # SET UP ONNX
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.inter_op_num_threads = 0
    if par:
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    ort_sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=opts)

    # SET UP ONNX ARGUMENTS
    state0 = ort.OrtValue.ortvalue_from_numpy(state0.numpy())
    args = { 'state': state0.numpy() }
    if gj:
        src = ort.OrtValue.ortvalue_from_numpy(gj_src.numpy())
        tgt = ort.OrtValue.ortvalue_from_numpy(gj_tgt.numpy())
        g_gj_arr = ort.OrtValue.ortvalue_from_numpy(np.array(g_gj, dtype='float32'))
        args['gj_src'] = src
        args['gj_tgt'] = tgt
        args['g_gj'] = g_gj_arr
    for arg_name, arg_value in cell_config:
        args[arg_name] = ort.OrtValue.ortvalue_from_numpy(arg_value)

    # PERFOMANCE MEASUREMENT
    T = .2
    measurements = []
    for _ in range(n_measurement_repititions):
        args['state'] = state0.numpy()
        niter = 0
        a = time.time() # WRONG PLACE
        while (b := time.time()) - a <= T:
            outputs = ort_sess.run(None, args)
            assert len(outputs) == 1
            args['state'] = outputs[0]
            niter += 1
        measurements.append(round(40000 / niter * (b - a), 3)) # Dont multiply by T but by elapsed time

    # RELATIVE ERROR MEASUREMENT
    args['state'] = state0.numpy()
    trace = []
    for _ in range(int(experiment_seconds * 1000 + .5)): # 1000 ms
        for _ in range(40): # 40 timesteps per ms
            args['state'] = ort_sess.run(None, args)[0]
        vsoma = args['state'][0, :]
        trace.append(vsoma)
    trace = np.array(trace)

    import matplotlib.pyplot as plt
    plt.plot(trace)
    plt.show()

    # RETURN
    return {
            'nneurons': nneurons,
            'par': par,
            'gj': gj,
            'runner': runner,
            'min': np.min(measurements),
            'max': np.max(measurements),
            'mean': np.mean(measurements),
            'std': np.std(measurements),
            'measurements': measurements,
            'g_gj': g_gj,
            'randomize_cell_params': randomize_cell_params,
            'trace': trace
    }

if __name__ == '__main__':
    print( run_experiment(4**3) )
