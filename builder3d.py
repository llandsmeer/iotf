import os
import numpy as np # for network generation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

NUM_STATE_VARS = 14
def make_initial_neuron_state(
        ncells,
        V_soma          = -60.0,
        soma_k          =   0.7423159,
        soma_l          =   0.0321349,
        soma_h          =   0.3596066,
        soma_n          =   0.2369847,
        soma_x          =   0.1,

        # Axon state
        V_axon          = -60.0,
        axon_Sodium_h   =   0.9,
        axon_Potassium_x=   0.2369847,

        # Dend state
        V_dend          = -60.0,
        dend_Ca2Plus    =   3.715,
        dend_Calcium_r  =   0.0113,
        dend_Potassium_s=   0.0049291,
        dend_Hcurrent_q =   0.0337836,
        dtype=tf.float32):

    # Soma State

    return tf.constant([

        # Soma state
        [V_soma]*ncells if V_soma is not None else np.random.normal(-60, 3, ncells),
        [soma_k]*ncells if soma_k is not None else np.random.random(ncells),
        [soma_l]*ncells if soma_l is not None else np.random.random(ncells),
        [soma_h]*ncells if soma_h is not None else np.random.random(ncells),
        [soma_n]*ncells if soma_n is not None else np.random.random(ncells),
        [soma_x]*ncells if soma_x is not None else np.random.random(ncells),

        # Axon state
        [V_axon]*ncells if V_axon is not None else np.random.normal(-60, 3, ncells),
        [axon_Sodium_h]*ncells if axon_Sodium_h is not None else np.random.random(ncells),
        [axon_Potassium_x]*ncells if axon_Potassium_x is not None else np.random.random(ncells),

        # Dend state
        [V_dend]*ncells if V_dend is not None else np.random.normal(-60, 3, ncells),
        [dend_Ca2Plus]*ncells,
        [dend_Calcium_r]*ncells if dend_Calcium_r is not None else np.random.random(ncells),
        [dend_Potassium_s]*ncells if dend_Potassium_s is not None else np.random.random(ncells),
        [dend_Hcurrent_q]*ncells if dend_Hcurrent_q is not None else np.random.random(ncells),

        ], dtype=dtype)

def sample_connections_3d(
        nneurons,
        nconnections=10,
        rmax=2,
        connection_probability=lambda r: np.exp(-(r/4)**2)):
    # we sample half the connections for each neuron
    assert nconnections % 2 == 0
    # we assume a cubic (4d toroid) brain
    nside = int(np.ceil(nneurons**(1/3)))
    if rmax > nside / 2: rmax = nside // 2
    # we set up a connection probability kernel around each neuron
    dx, dy, dz = np.mgrid[-rmax:rmax+1, -rmax:rmax+1, -rmax:rmax+1]
    dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    # we only sample backwards, as the forward connections
    # are part of the kernel of other neurons
    sample_backwards = \
            ((dz < 0)) | \
            ((dz == 0) &( dy < 0)) | \
            ((dz == 0) & (dy == 0) & (dx < 0))
    m = (r != 0) & sample_backwards & (r < rmax)
    dx, dy, dz, r = dx[m], dy[m], dz[m], r[m]
    P = connection_probability(r)

    # next, there is a ~r^2 increase in point density per r,
    # and very non uniform distribution of those due to
    # the integer grid. let's remove that bias
    _, r_uniq_idx = np.unique(r, return_inverse=True)
    r_idx_freq = np.bincount(r_uniq_idx)
    r_freq = r_idx_freq[r_uniq_idx]
    P = P / r_freq
    # P must sum up to 1
    P = P / P.sum()

    # a connection connects two neurons
    final_connection_count = nneurons * nconnections // 2

    # instead of sampling using the P array,
    # we sample for each value of the P array,
    # which is much more memory efficient
    counts = (P * final_connection_count + .5).astype(int)
    counts[-1] =  max(0, final_connection_count - counts[:-1].sum())
    assert (counts < nneurons).all()
    conn_idx = []
    for draw in range(len(P)):
        if counts[draw] == 0:
            continue
        if counts[draw] == 1:
            draw_idx = np.array([np.random.randint(nneurons)])
        else:
            draw_idx = np.random.choice(nneurons, counts[draw], replace=False)
        conn_idx.append(draw + len(P) * draw_idx)
    conn_idx = np.concatenate(conn_idx)

    # now we calculate the neuron indices back from the P kernel
    neuron_id1 = conn_idx // len(P)
    x = ( neuron_id1 %  nside).astype('int32')
    y = ((neuron_id1 // nside) % nside).astype('int32')
    z = ((neuron_id1 // (nside*nside)) % nside).astype('int32')

    di = conn_idx % len(P)

    neuron_id2 = ( \
        (x + dx[di]) % nside + \
        (y + dy[di]) % nside * nside + \
        (z + dz[di]) % nside * nside * nside
        ).astype(int)

    # and generate the final index arrays
    # needed for gj calculation
    tgt_idx = np.concatenate([neuron_id1, neuron_id2])
    src_idx = np.concatenate([neuron_id2, neuron_id1])

    return tf.constant(src_idx, dtype='int32'), \
           tf.constant(tgt_idx, dtype='int32')

def timestep(
        state,

        # Simulation parameters
        delta=0.025,

        # Geometry parameters
        g_int           =   0.13,    # Cell internal conductance  -- now a parameter (0.13)
        p1              =   0.25,    # Cell surface ratio soma/dendrite
        p2              =   0.15,    # Cell surface ratio axon(hillock)/soma

        # Channel conductance parameters
        g_CaL           =   1.1,     # Calcium T - (CaV 3.1) (0.7)
        g_h             =   0.12,    # H current (HCN) (0.4996)
        g_K_Ca          =  35.0,     # Potassium  (KCa v1.1 - BK) (35)
        g_ld            =   0.01532, # Leak dendrite (0.016)
        g_la            =   0.016,   # Leak axon (0.016)
        g_ls            =   0.016,   # Leak soma (0.016)
        g_Na_s          = 150.0,     # Sodium  - (Na v1.6 )
        g_Kdr_s         =   9.0,     # Potassium - (K v4.3)
        g_K_s           =   5.0,     # Potassium - (K v3.4)
        g_CaH           =   4.5,     # High-threshold calcium -- Ca V2.1
        g_Na_a          = 240.0,     # Sodium
        g_K_a           = 240.0,     # Potassium (20)

        # Membrane capacitance
        S               =   1.0,     # 1/C_m, cm^2/uF

        # Reversal potential parameters
        V_Na            =  55.0,     # Sodium
        V_K             = -75.0,     # Potassium
        V_Ca            = 120.0,     # Low-threshold calcium channel
        V_h             = -43.0,     # H current
        V_l             =  10.0,     # Leak

        # Stimulus parameter
        I_app           =   0.0,
        gj_src          = None,
        gj_tgt          = None,
        g_gj            = 0.05
        ):

    assert state.shape[0] == NUM_STATE_VARS

    # Soma state
    V_soma              = state[0, :]
    soma_k              = state[1, :]
    soma_l              = state[2, :]
    soma_h              = state[3, :]
    soma_n              = state[4, :]
    soma_x              = state[5, :]

    # Axon state
    V_axon              = state[6, :]
    axon_Sodium_h       = state[7, :]
    axon_Potassium_x    = state[8, :]

    # Dend state
    V_dend              = state[9, :]
    dend_Ca2Plus        = state[10,:]
    dend_Calcium_r      = state[11,:]
    dend_Potassium_s    = state[12,:]
    dend_Hcurrent_q     = state[13,:]

    ########## SOMA UPDATE ##########

    # CURRENT: Soma leak current (ls)
    soma_I_leak        = g_ls * (V_soma - V_l)

    # CURRENT: Soma interaction current (ds, as)
    I_ds        =  (g_int / p1)        * (V_soma - V_dend)
    I_as        =  (g_int / (1 - p2))  * (V_soma - V_axon)
    soma_I_interact =  I_ds + I_as

    # CHANNEL: Soma Low-threshold calcium (CaL)
    soma_Ical   = g_CaL * soma_k * soma_k * soma_k * soma_l * (V_soma - V_Ca)

    soma_k_inf  = 1 / (1 + tf.exp(-(V_soma + 61)/4.2))
    soma_l_inf  = 1 / (1 + tf.exp( (V_soma + 85)/8.5))
    soma_tau_l  = (20 * tf.exp((V_soma + 160)/30) / (1 + tf.exp((V_soma + 84) / 7.3))) + 35

    soma_dk_dt  = soma_k_inf - soma_k
    soma_dl_dt  = (soma_l_inf - soma_l) / soma_tau_l

    # CHANNEL: Soma sodium (Na_s)
    # watch out direct gate: m = m_inf
    soma_m_inf  = 1 / (1 + tf.exp(-(V_soma + 30)/5.5))
    soma_h_inf  = 1 / (1 + tf.exp( (V_soma + 70)/5.8))
    soma_Ina    = g_Na_s * soma_m_inf**3 * soma_h * (V_soma - V_Na)
    soma_tau_h  = 3 * tf.exp(-(V_soma + 40)/33)
    soma_dh_dt  = (soma_h_inf - soma_h) / soma_tau_h

    # CHANNEL: Soma potassium, slow component (Kdr)
    soma_Ikdr   = g_Kdr_s * soma_n**4 * (V_soma - V_K)
    soma_n_inf  = 1 / ( 1 + tf.exp(-(V_soma +  3)/10))
    soma_tau_n  = 5 + (47 * tf.exp( (V_soma + 50)/900))
    soma_dn_dt  = (soma_n_inf - soma_n) / soma_tau_n

    # CHANNEL: Soma potassium, fast component (K_s)
    soma_Ik      = g_K_s * soma_x**4 * (V_soma - V_K)
    soma_alpha_x = 0.13 * (V_soma + 25) / (1 - tf.exp(-(V_soma + 25)/10))
    soma_beta_x  = 1.69 * tf.exp(-(V_soma + 35)/80)
    soma_tau_x_inv=soma_alpha_x + soma_beta_x
    soma_x_inf   = soma_alpha_x / soma_tau_x_inv

    soma_dx_dt   = (soma_x_inf - soma_x) * soma_tau_x_inv

    # UPDATE: Soma compartment update (V_soma)
    soma_I_Channels = soma_Ik + soma_Ikdr + soma_Ina + soma_Ical
    soma_dv_dt = S * (-(soma_I_leak + soma_I_interact + soma_I_Channels))

    ########## AXON UPDATE ##########

    # CURRENT: Axon leak current (la)
    axon_I_leak    =  g_la * (V_axon - V_l)

    # CURRENT: Axon interaction current (sa)
    I_sa           =  (g_int / p2) * (V_axon - V_soma)
    axon_I_interact=  I_sa

    # CHANNEL: Axon sodium (Na_a)
    # watch out direct gate: m = m_inf
    axon_m_inf     =  1 / (1 + tf.exp(-(V_axon+30)/5.5))
    axon_h_inf     =  1 / (1 + tf.exp( (V_axon+60)/5.8))
    axon_Ina       =  g_Na_a * axon_m_inf**3 * axon_Sodium_h * (V_axon - V_Na)
    axon_tau_h     =  1.5 * tf.exp(-(V_axon+40)/33)
    axon_dh_dt     =  (axon_h_inf - axon_Sodium_h) / axon_tau_h

    # CHANNEL: Axon potassium (K_a)
    axon_Ik        =  g_K_a * axon_Potassium_x**4 * (V_axon - V_K)
    axon_alpha_x   =  0.13*(V_axon + 25) / (1 - tf.exp(-(V_axon + 25)/10))
    axon_beta_x    =  1.69 * tf.exp(-(V_axon + 35)/80)
    axon_tau_x_inv =  axon_alpha_x + axon_beta_x
    axon_x_inf     =  axon_alpha_x / axon_tau_x_inv
    axon_dx_dt     =  (axon_x_inf - axon_Potassium_x) * axon_tau_x_inv

    # UPDATE: Axon hillock compartment update (V_axon)
    axon_I_Channels = axon_Ina + axon_Ik
    axon_dv_dt  = S * (-(axon_I_leak +  axon_I_interact + axon_I_Channels))

    ########## DEND UPDATE ##########

    # CURRENT: Dend application current (I_app)

    if gj_src is not None and gj_tgt is not None:
        vdiff = tf.gather(V_dend, gj_src) - tf.gather(V_dend, gj_tgt)
        cx36_current_per_gj = (0.2 + 0.8 * tf.exp(-vdiff*vdiff / 100)) * vdiff * g_gj
        I_gapp = tf.tensor_scatter_nd_add(tf.zeros_like(V_dend), tf.reshape(gj_tgt, (-1, 1)),
            cx36_current_per_gj)
    else:
        I_gapp = 0

    dend_I_application = -I_app - I_gapp

    # CURRENT: Dend leak current (ld)
    dend_I_leak     =  g_ld * (V_dend - V_l)

    # CURRENT: Dend interaction Current (sd)
    dend_I_interact =  (g_int / (1 - p1)) * (V_dend - V_soma)

    # CHANNEL: Dend high-threshold calcium (CaH)
    dend_Icah       =  g_CaH * dend_Calcium_r * dend_Calcium_r * (V_dend - V_Ca)
    dend_alpha_r    =  1.7 / (1 + tf.exp(-(V_dend - 5)/13.9))
    dend_beta_r     =  0.02*(V_dend + 8.5) / (tf.exp((V_dend + 8.5)/5) - 1.0)
    dend_tau_r_inv5 =  (dend_alpha_r + dend_beta_r) # tau = 5 / (alpha + beta)
    dend_r_inf      =  dend_alpha_r / dend_tau_r_inv5
    dend_dr_dt      =  (dend_r_inf - dend_Calcium_r) * dend_tau_r_inv5 * 0.2

    # CHANNEL: Dend calcium dependent potassium (KCa)
    dend_Ikca       =  g_K_Ca * dend_Potassium_s * (V_dend - V_K)
    dend_alpha_s    =  tf.where(
            0.00002 * dend_Ca2Plus < 0.01,
            0.00002 * dend_Ca2Plus,
            0.01)
    dend_tau_s_inv  =  dend_alpha_s + 0.015
    dend_s_inf      =  dend_alpha_s / dend_tau_s_inv
    dend_ds_dt      =  (dend_s_inf - dend_Potassium_s) * dend_tau_s_inv

    # CHANNEL: Dend proton (h)
    dend_Ih         =  g_h * dend_Hcurrent_q * (V_dend - V_h)
    q_inf           =  1 / (1 + tf.exp((V_dend + 80)/4))
    tau_q_inv       =  tf.exp(-0.086*V_dend - 14.6) + tf.exp(0.070*V_dend - 1.87)
    dend_dq_dt      =  (q_inf - dend_Hcurrent_q) * tau_q_inv

    # CONCENTRATION: Dend calcium concentration (CaPlus)
    dend_dCa_dt          =  -3 * dend_Icah - 0.075 * dend_Ca2Plus

    # UPDATE: Dend compartment update (V_dend)
    dend_I_Channels = dend_Icah + dend_Ikca + dend_Ih
    dend_dv_dt  = S * (-(dend_I_leak +  dend_I_interact + dend_I_application + dend_I_Channels))

    ########## UPDATE ##########

    return tf.stack([
        # Soma state
        V_soma              + soma_dv_dt * delta,
        soma_k              + soma_dk_dt * delta,
        soma_l              + soma_dl_dt * delta,
        soma_h              + soma_dh_dt * delta,
        soma_n              + soma_dn_dt * delta,
        soma_x              + soma_dx_dt * delta,
        # Axon state
        V_axon              + axon_dv_dt * delta,
        axon_Sodium_h       + axon_dh_dt * delta,
        axon_Potassium_x    + axon_dx_dt * delta,
        # Dend state
        V_dend              + dend_dv_dt * delta,
        dend_Ca2Plus        + dend_dCa_dt* delta,
        dend_Calcium_r      + dend_dr_dt * delta,
        dend_Potassium_s    + dend_ds_dt * delta,
        dend_Hcurrent_q     + dend_dq_dt * delta,
        ], axis=0)

def make_function(*, ngj=0, ncells=None, argconfig=()):
    MAKE_FUNCTION_TEMPLATE = '@tf.function\ndef wrapper({function_args}): return timestep({call_args})'
    import io
    argconfig = dict(argconfig)
    cell_params = ['g_int', 'p1', 'p2', 'g_CaL', 'g_h', 'g_K_Ca',
                  'g_ld', 'g_la', 'g_ls', 'g_Na_s', 'g_Kdr_s', 'g_K_s',
                  'g_CaH', 'g_Na_a', 'g_K_a', 'V_Na', 'V_K', 'V_Ca',
                  'V_h', 'V_l', 'I_app', 'delta' ]
    function_args = ['state'] # generated function signature in python
    call_args = ['state'] # call arguments to timestep()
    argspec = [tf.TensorSpec((NUM_STATE_VARS, ncells), tf.float32, name='state')] # TensorFlow argspec
    for param in cell_params:
        if param not in argconfig:
            continue
        value = argconfig.pop(param)
        if isinstance(value, (float, int)):
            call_args.append(f'{param}={value}')
        elif value == 'CONSTANT':
            function_args.append(param)
            call_args.append(f'{param}={param}')
            argspec.append(tf.TensorSpec((), tf.float32, name=param))
        elif value == 'VARY':
            function_args.append(param)
            call_args.append(f'{param}={param}')
            argspec.append(tf.TensorSpec((ncells,), tf.float32, name=param))
        else:
            raise ValueError(f'Unknown argconfig {param}={repr(value)}. Must be float, int, "CONSTANT" or "VARY"')
    if argconfig:
        raise ValueError(f'Leftover argconfig {argconfig}')
    if ngj != 0:
        for arg in 'gj_src', 'gj_tgt', 'g_gj':
            function_args.append(arg)
            call_args.append(f'{arg}={arg}')
        argspec.append(tf.TensorSpec(ngj, tf.int32, name='gj_src'))
        argspec.append(tf.TensorSpec(ngj, tf.int32, name='gj_tgt'))
        argspec.append(tf.TensorSpec((), tf.float32, name='g_gj'))

    function_args = ', '.join(function_args)
    call_args = ', '.join(call_args)

    final = MAKE_FUNCTION_TEMPLATE.format(function_args=function_args, call_args=call_args)
    env = dict(
        tf = tf,
        timestep=timestep
    )
    exec(final, env)
    wrapper = env['wrapper']
    wrapper.argspec = argspec
    return wrapper

def test_onnx():
    import tf2onnx
    import inspect
    print('#'*100)
    print('MAKING FUNCTION'.center(100))
    print('#'*100)
    tf_function = make_function(
        ngj=1000,
        ncells=1000,
        argconfig=dict(
        g_CaL = 1.0,
        p1 = 'CONSTANT'
        ))
    print('#'*100)
    print('CONVERTING TO ONNX'.center(100))
    print('#'*100)
    onnx_model, _ = tf2onnx.convert.from_function(
            function=tf_function, 
            input_signature=tf_function.argspec,
            output_path='/tmp/io2.onnx',
            opset=16,
            )

    # print('#'*100)
    # print('CONVERTING TO KERAS'.center(100))
    # print('#'*100)
    # from onnx2keras import onnx_to_keras
    # keras_model = onnx_to_keras(onnx_model, [arg.name for arg in tf_function.argspec])
    # print(keras_model)
    # print('#'*100)

def test_onnx_max():
    import tf2onnx
    import inspect
    for ncells in 10, 100, 1000, 10000, 100000:
        print('#'*100)
        print('MAKING FUNCTION'.center(100))
        print('#'*100)
        tf_function = make_function(
            ngj=ncells*10,
            ncells=ncells,
            argconfig=dict(
            g_CaL = 1.0,
            p1 = 'CONSTANT'
            ))
        print('#'*100)
        print('CONVERTING TO ONNX'.center(100))
        print('#'*100)
        onnx_model, _ = tf2onnx.convert.from_function(
                function=tf_function, 
                input_signature=tf_function.argspec,
                output_path=f'/tmp/io{ncells}.onnx',
                opset=16,
                )


def test_simple():
    import matplotlib.pyplot as plt
    nneurons = 5**3
    gj_src, gj_tgt = sample_connections_3d(nneurons//2)
    out = []
    state = make_initial_neuron_state(nneurons, dtype=tf.float32, V_axon=None, V_dend=None, V_soma=None)
    for i in range(10000):
        state = timestep(state, gj_src=gj_src, gj_tgt=gj_tgt)
        if i % 100 == 0:
            out.append(np.array(state[0]))
    out = np.array(out)
    plt.plot(out)
    plt.show()



if __name__ == '__main__':
    test_onnx_max()

