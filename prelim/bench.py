import sys
sys.path.append('..')
import subprocess
import json
import socket
import time
import numpy as np
import tensorflow as tf

if len(sys.argv) != 2:
    print('usage: bench.py [cpu|gpu|groq]')
    exit(1)
else:
    mode = sys.argv[1]
    assert mode in ('cpu', 'gpu', 'groq')

import ioperf

LIF_NUM_STATE_VARS = 2
HH_NUM_STATE_VARS = 5

def lif_make_initial(ncells, V=None):
    return tf.constant([
        [0]*ncells if V is not None else np.random.normal(0, 3, ncells),
        [0]*ncells
        ], dtype=tf.float32)
def lif_make_timestep40(ncells, nconns):
    argspec =[tf.TensorSpec((LIF_NUM_STATE_VARS, ncells), tf.float32, name='state'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='V_th'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='delta'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='tau_syn'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='tau_mem'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='iint'),
              tf.TensorSpec((nconns,), tf.int32, name='spike_src'),
              tf.TensorSpec((nconns,), tf.int32, name='spike_tgt'),
              tf.TensorSpec((), tf.float32, name='spike_w'),
              ]
    @tf.function(input_signature=argspec, jit_compile=True)
    def timestep(
            state,
            spike_src          = None,
            spike_tgt          = None,
            spike_w            = 0.05,
            ):
        #CONSTANT:
        V_th               = 50;
        delta              = 0.025;
        tau_syn            = 10.;
        tau_mem            = 4.;
        iint               = 3.0;
        assert state.shape[0] == LIF_NUM_STATE_VARS
        V              = state[0, :]
        Isyn           = state[1, :]
        alpha          = tf.exp(-delta / tau_syn)
        beta           = tf.exp(-delta / tau_mem)
        S = V >= V_th
        #recv = tf.gather(spike_tgt, tf.where(tf.gather(S, spike_src))[:,0])
        #w = tf.cast(recv * 0, tf.float32) + spike_w
        #syn_in = tf.tensor_scatter_nd_add(tf.zeros_like(Isyn), tf.reshape(recv, (-1, 1)), w)
        recv = tf.where(tf.gather(S, spike_src), spike_w, 0.0)
        syn_in = tf.tensor_scatter_nd_add(tf.zeros_like(Isyn), tf.reshape(spike_tgt, (-1, 1)), recv)
        V_next = tf.where(S, 0., beta * V + Isyn + iint)
        Isyn_next = Isyn * alpha + syn_in
        state_next = tf.stack([V_next, Isyn_next], axis=0)
        return {"state_next": state_next } #, 'S': tf.math.count_nonzero(S)}
    @tf.function(input_signature=argspec, jit_compile=True)
    def timestep40(state,
                   #CONSTANT: V_th, delta, tau_syn, tau_mem, iint,
                   spike_src, spike_tgt, spike_w):
        S = 0
        for _ in range(40):
            out = timestep(state,
                           #CONSTANT: V_th=V_th, delta=delta, tau_syn=tau_syn, tau_mem=tau_mem, iint=iint,
                           spike_src=spike_src, spike_tgt=spike_tgt, spike_w=spike_w)
            state = out['state_next']
            # S = S + out['S']
        return {"state_next": state }#, 'S': S}
    return timestep40
def lif_timeit(ncells):
    spike_src, spike_tgt = ioperf.model.sample_connections_3d(ncells, rmax=4)
    spike_src_asym = spike_src[:len(spike_src)//2]
    spike_tgt_asym = spike_tgt[:len(spike_tgt)//2]
    state = lif_make_initial(ncells)
    lif40 = lif_make_timestep40(ncells, len(spike_src_asym))
    burn_in = 50
    args = (
        #CONSTANT: tf.constant(50, dtype=tf.float32),
        #CONSTANT: tf.constant(0.025, dtype=tf.float32), #delta
        #CONSTANT: tf.constant(10, dtype=tf.float32), #tau
        #CONSTANT: tf.constant(4, dtype=tf.float32), #tau
        #CONSTANT: tf.constant(3, dtype=tf.float32), #iint
        spike_src_asym,
        spike_tgt_asym,
        tf.constant(0.05, dtype=tf.float32) # wint
        )
    for i in range(burn_in):
        state_next = lif40(state, *args)
        state = state_next['state_next']
    ms = 1000
    spike_count = 0
    a = time.perf_counter()
    for i in range(ms):
        state_next = lif40(state, *args)
        # spike_count += state_next['S'].numpy()
        state = state_next['state_next']
    b = time.perf_counter()
    f = spike_count / ncells / ms * 1e3
    elapsed = b - a
    # print('>'*10, 'firing frequency', f)
    print('>'*10, 'seconds/second', elapsed)
    return elapsed

# timeit(8**3)


@tf.function(jit_compile=True)
def exprelr(x): return x / (tf.exp(x)-1)
@tf.function(jit_compile=True)
def alpha_m(V): return exprelr(-0.1*V - 4.0)
@tf.function(jit_compile=True)
def alpha_h(V): return 0.07*tf.exp(-0.05*V - 3.25)
@tf.function(jit_compile=True)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
@tf.function(jit_compile=True)
def beta_m(V): return 4.0*tf.exp(-(V + 65.0)/18.0)
@tf.function(jit_compile=True)
def beta_h(V): return 1.0/(tf.exp(-0.1*V - 3.5) + 1.0)
@tf.function(jit_compile=True)
def beta_n(V): return 0.125*tf.exp(-0.0125*V - 0.8125)
def hh_make_initial(ncells):
    V = tf.constant(np.random.normal(-60, 3, ncells))
    m = (alpha_m(V) / (alpha_m(V) + beta_m(V))).numpy()
    h = (alpha_h(V) / (alpha_h(V) + beta_h(V))).numpy()
    n = (alpha_n(V) / (alpha_n(V) + beta_n(V))).numpy()
    return tf.constant([
        V.numpy().tolist(), m.tolist(), h.tolist(), n.tolist(), [0]*ncells
        ], dtype=tf.float32)
def hh_make_timestep40(ncells, nconns):
    argspec =[tf.TensorSpec((HH_NUM_STATE_VARS, ncells), tf.float32, name='state'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='V_th'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='delta'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='g_leak'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='g_na'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='g_k'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='E_leak'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='E_na'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='E_k'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='S'),
              #CONSTANT: tf.TensorSpec((), tf.float32, name='tau_syn'),
              tf.TensorSpec((ncells,), tf.float32, name='iint'),
              tf.TensorSpec((nconns,), tf.int32, name='spike_src'),
              tf.TensorSpec((nconns,), tf.int32, name='spike_tgt'),
              tf.TensorSpec((), tf.float32, name='spike_w'),
              ]
    @tf.function(input_signature=argspec, jit_compile=True)
    def timestep(
            state,
            iint               = 0.0,
            spike_src          = None,
            spike_tgt          = None,
            spike_w            = 0.05,
            ):
        #CONSTANT:
        V_th               = -10;
        delta              = 0.025;
        g_na = 120; g_k = 36; g_leak = 0.3; E_na = 55; E_k = -77; E_leak = -65;
        S                  = 1.;
        tau_syn            = 1.;
        #CONSTANT:
        assert state.shape[0] == HH_NUM_STATE_VARS
        V              = state[0, :]
        m              = state[1, :]
        h              = state[2, :]
        n              = state[3, :]
        Isyn           = state[4, :]
        I_Na = g_na*m**3*h*(V - E_na)
        I_K = g_k*n**4*(V - E_k)
        I_L = g_leak*(V - E_leak)
        dm_dt = alpha_m(V)*(1-m) - beta_m(V)*m
        dh_dt = alpha_h(V)*(1-h) - beta_h(V)*h
        dn_dt = alpha_n(V)*(1-n) - beta_n(V)*n
        alpha = tf.exp(-delta / tau_syn)
        dv_dt = -(I_Na + I_K + I_L - iint - Isyn) * S
        V_next = V + delta * dv_dt
        spike_flag = (V < V_th) & (V_next >= V_th)
        # ~4x slower
        # recv = tf.gather(spike_tgt, tf.where(tf.gather(spike_flag, spike_src))[:,0])
        # w = tf.cast(recv * 0, tf.float32) + spike_w
        # syn_in = tf.tensor_scatter_nd_add(tf.zeros_like(Isyn), tf.reshape(recv, (-1, 1)), w)
        #
        recv = tf.where(tf.gather(spike_flag, spike_src), spike_w, 0.0)
        syn_in = tf.tensor_scatter_nd_add(tf.zeros_like(Isyn), tf.reshape(spike_tgt, (-1, 1)), recv)
        #
        # syn_in = 0
        #
        Isyn_next = Isyn * alpha + syn_in
        state_next = tf.stack([
            V_next,
            m + delta * dm_dt,
            h + delta * dh_dt,
            n + delta * dn_dt,
            Isyn_next
            ], axis=0)
        return {"state_next": state_next} #, 'S': tf.math.count_nonzero(spike_flag)}
    @tf.function(input_signature=argspec, jit_compile=True)
    def timestep40(state,
                   #CONSTANT: V_th, delta, g_na, g_k, g_leak, E_na, E_k, E_leak, S, tau_syn,
                   iint, spike_src, spike_tgt, spike_w):
        spike_count = 0
        for _ in range(40):
            out = timestep(state=state,
                           #CONSTANT: V_th=V_th, delta=delta, g_na=g_na, g_k=g_k, g_leak=g_leak, E_na=E_na, E_k=E_k, E_leak=E_leak, S=S, tau_syn=tau_syn,
                           iint=iint, spike_src=spike_src, spike_tgt=spike_tgt, spike_w=spike_w)
            state = out['state_next']
            # spike_count = spike_count + out['S']
        return {"state_next": state}#, 'S': spike_count}
    return timestep40
def hh_timeit(ncells):
    spike_src, spike_tgt = ioperf.model.sample_connections_3d(ncells, rmax=4)
    spike_src_asym = spike_src[:len(spike_src)//2]
    spike_tgt_asym = spike_tgt[:len(spike_tgt)//2]
    state = hh_make_initial(ncells)
    print(state.shape, spike_src_asym.shape)
    hh40 = hh_make_timestep40(ncells, len(spike_src_asym))
    burn_in = 50
    vs = []
    args = dict(
            #CONSTANT: V_th=tf.constant(-10, dtype=tf.float32), delta=tf.constant(0.025, dtype=tf.float32),
            #CONSTANT: g_na = tf.constant(120, dtype=tf.float32), g_k = tf.constant(36, dtype=tf.float32), g_leak = tf.constant(0.3, dtype=tf.float32),
            #CONSTANT: E_na = tf.constant(55, dtype=tf.float32), E_k = tf.constant(-77., dtype=tf.float32), E_leak = tf.constant(-65, dtype=tf.float32),
            #CONSTANT: S                  = tf.constant(1.0, dtype=tf.float32),
            #CONSTANT: tau_syn            = tf.constant(1.0, dtype=tf.float32),
            spike_src          = spike_src_asym,
            spike_tgt          = spike_tgt_asym,
            spike_w            = tf.constant(0.05, dtype=tf.float32)
                )
    for i in range(burn_in):
        args['iint'] = tf.constant(np.random.random(ncells)**5*10, dtype=tf.float32)
        state_next = hh40(state, **args)
        # vs.append(state.numpy()[0, :])
        state = state_next['state_next']
    ms = 1000
    spike_count = 0
    iints = [tf.constant(np.random.random(ncells)**5*10, dtype=tf.float32) for _ in range(ms)]
    a = time.perf_counter()
    for i in range(ms):
        args['iint'] = iints[i]
        state_next = hh40(state, **args)
        # spike_count += state_next['S'].numpy()
        # vs.append(state.numpy()[0, :])
        state = state_next['state_next']
    # vs.append(state.numpy()[0, :])
    b = time.perf_counter()
    f = spike_count / ncells / ms * 1e3
    elapsed = b - a
    print('>'*10, 'firing frequency', f)
    print('>'*10, 'seconds/second', elapsed)
    #import matplotlib.pyplot as plt
    #plt.plot(np.array(vs))
    #print(vs[-1])
    #plt.show()
    return elapsed

def io_timeit(ncells):
    src, tgt = ioperf.model.sample_connections_3d(ncells, rmax=4)
    state = ioperf.model.make_initial_neuron_state(ncells, V_soma=None)
    print(state.shape, src.shape)
    argconfig = dict( I_app='VARY', g_CaL='VARY' )
    timestep40 = ioperf.model.make_tf_function_40(ngj=len(src), argconfig=argconfig)
    burn_in = 50
    I_app = np.zeros(ncells, dtype='float32')
    g_CaL = np.array(0.5+0.9*np.random.random(ncells), dtype='float32')
    for i in range(burn_in):
        state = timestep40(state, gj_src=src, gj_tgt=tgt, g_gj=0.05, I_app=I_app, g_CaL=g_CaL)['state_next']
    ms = 1000
    a = time.perf_counter()
    for i in range(ms):
        state = timestep40(state, gj_src=src, gj_tgt=tgt, g_gj=0.05, I_app=I_app, g_CaL=g_CaL)['state_next']
    b = time.perf_counter()
    elapsed = b - a
    print('>'*10, 'seconds/second', elapsed)
    return elapsed

def log(**kw):
    print('LOG')
    kw['ctime'] = time.ctime()
    for k, v in kw.items():
        print('    ', k.ljust(20), v)
        kw[k] = v
    with open('log.json', 'a') as f:
        print(json.dumps(kw), file=f)


ns = [4**3, 5**3, 6**3, 7**3, 8**3, 9**3, 10**3, 20**3, 30**3, 40**3, 50**3, 60**3, 70**3, 80**3, 90**3, 100**3]

log(mode=mode, host=socket.gethostname(), commit=subprocess.getoutput('git rev-parse --short HEAD'))
if mode == 'gpu':
    assert len(tf.config.list_physical_devices('GPU')) > 0
    out = []
    with tf.device('/GPU:0'):
        for n in ns:
            print(n)
            a = lif_timeit(n)
            b = hh_timeit(n) # 18 Hz
            c = io_timeit(n)
            log(n=n, lif=a, hh=b, io=c)
elif mode == 'cpu':
    out = []
    with tf.device('/CPU:0'):
        for n in ns:
            print(n)
            a = lif_timeit(n)
            b = hh_timeit(n) # 18 Hz
            c = io_timeit(n)
            log(n=n, lif=a, hh=b, io=c)
# elif mode == 'groq':
#     import tempfile
#     import tf2onnx
# 
#     tf_function = model.make_tf_function_40(*args, **kwargs)
#     tf_function = make_hh
# 
#     path = tempfile.mktemp() + '.onnx'
# 
#     onnx_model, _ = tf2onnx.convert.from_function(
#             function=tf_function,
#             input_signature=tf_function.argspec,
#             output_path=path,
#             opset=opset,
#             )
# 
#     return path
