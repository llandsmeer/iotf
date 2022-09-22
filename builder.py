import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

NUM_STATE_VARS = 14
def make_initial_neuron_state(ncells, dtype=tf.float32):

    # Soma State
    V_soma          = -60.0
    soma_k          =   0.7423159
    soma_l          =   0.0321349
    soma_h          =   0.3596066
    soma_n          =   0.2369847
    soma_x          =   0.1

    # Axon state
    V_axon          = -60.0
    axon_Sodium_h   =   0.9
    axon_Potassium_x=   0.2369847

    # Dend state
    V_dend          = -60.0
    dend_Ca2Plus    =   3.715
    dend_Calcium_r  =   0.0113
    dend_Potassium_s=   0.0049291
    dend_Hcurrent_q =   0.0337836

    return tf.constant([[

        # Soma state
        [V_soma]*ncells,
        [soma_k]*ncells,
        [soma_l]*ncells,
        [soma_h]*ncells,
        [soma_n]*ncells,
        [soma_x]*ncells,

        # Axon state
        [V_axon]*ncells,
        [axon_Sodium_h]*ncells,
        [axon_Potassium_x]*ncells,

        # Dend state
        [V_dend]*ncells,
        [dend_Ca2Plus]*ncells,
        [dend_Calcium_r]*ncells,
        [dend_Potassium_s]*ncells,
        [dend_Hcurrent_q]*ncells,

        ]], dtype=dtype)

def make_sparse_matrices_for_gap_junctions(pairs, ncells=0):
    import numpy as np # ONLY user here for float32 conversion
    # tensorflow sparse array is constructed COO
    # a dictionary mapping indices (i, j) to values (v)
    # which we give as two separate lists
    #
    # scatter is maps the voltage array to an voltage difference per gap junction array
    scatter_indices = []
    scatter_values = []
    # gather maps the conductances, derived from the voltage differences to currents
    gather_indices = []
    gather_values = []
    for idx, (w, i, j) in enumerate(pairs):
        i, j = sorted([i, j])
        # if ncells was not given or too small, update to maximum cell index
        ncells = max(ncells, i+1, j+1)
        # bidirectional connections
        for gj_id, self, other in [2*idx, i, j], [2*idx+1, j, i]:
            # scatter: vdiff[gj_id] = v[other] - v[self]
            scatter_indices.extend([(gj_id, self), (gj_id, other)])
            scatter_values.extend([1, -1])
            # gather: i[self] = sum_{gj_ids(self)} w * f(vdiff[gj_id])
            gather_indices.append((self, gj_id))
            gather_values.append(w)
    scatter = tf.SparseTensor(
            indices=np.array(scatter_indices, dtype='float32'),
            values=np.array(scatter_values, dtype='float32'),
            dense_shape=(len(pairs)*2, ncells)
            )
    gather = tf.SparseTensor(
            indices=np.array(gather_indices, dtype='float32'),
            values=np.array(gather_values, dtype='float32'),
            dense_shape=(ncells, len(pairs)*2)
            )
    return scatter, gather

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
        gj_scatter      = None,
        gj_gather       = None,
        ):

    assert state.shape[0] == NUM_STATE_VARS

    # Soma state
    V_soma              = state[:, 0]
    soma_k              = state[:, 1]
    soma_l              = state[:, 2]
    soma_h              = state[:, 3]
    soma_n              = state[:, 4]
    soma_x              = state[:, 5]

    # Axon state
    V_axon              = state[:, 6]
    axon_Sodium_h       = state[:, 7]
    axon_Potassium_x    = state[:, 8]

    # Dend state
    V_dend              = state[:, 9]
    dend_Ca2Plus        = state[:,10]
    dend_Calcium_r      = state[:,11]
    dend_Potassium_s    = state[:,12]
    dend_Hcurrent_q     = state[:,13]

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

    if gj_scatter is not None and gj_gather is not None:
        gj_vdiff = tf.sparse.sparse_dense_matmul(gj_scatter, V_dend[..., None])
        gj_nonlin = (0.2 + 0.8 * tf.exp(-0.01 * gj_vdiff*gj_vdiff)) * gj_vdiff
        I_gapp = tf.sparse.sparse_dense_matmul(gj_gather, gj_nonlin)[..., 0]
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
        ], axis=1)

# def build_function_spec(fixed=(), variable=()):
#     fixed = set(fixed)
#     variable = set(variable)
#     spec = []
#     for param in list(inspect.signature(timestep).parameters.values()):
#         if param.default == param.empty:
#             assert param.name == 'state'
#         if param.name == 'state':
#             spec.append(tf.TensorSpec((14, ncells), tf.float32, name='state'))
#         elif param.name in fixed:
#             fixed.remove(param.name)
#             spec.append(tf.TensorSpec((), tf.float32, name=param.name))
#         elif param.name in variable:
#             variable.remove(param.name)
#             spec.append(tf.TensorSpec((ncells,), tf.float32, name=param.name))
#         else:
#             spec.append(tf.TensorSpec((ncells,), tf.float32, name=param.name))
#     assert not fixed
#     assert not variable
#     return spec

def make_function(*, gj=False, ncells=None, argconfig=()):
    MAKE_FUNCTION_TEMPLATE = '@tf.function\ndef wrapper({function_args}): return timestep({call_args})'
    import io
    argconfig = dict(argconfig)
    cell_params = ['g_int', 'p1', 'p2', 'g_CaL', 'g_h', 'g_K_Ca',
                  'g_ld', 'g_la', 'g_ls', 'g_Na_s', 'g_Kdr_s', 'g_K_s',
                  'g_CaH', 'g_Na_a', 'g_K_a', 'V_Na', 'V_K', 'V_Ca',
                  'V_h', 'V_l', 'I_app', 'delta' ]
    gj_params = ['gj_scatter', 'gj_gather']
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
    if gj:
        function_args.append('gj_scatter')
        function_args.append('gj_gather')
        call_args.append('gj_scatter=gj_scatter')
        call_args.append('gj_gather=gj_gather')
        spec = tf.SparseTensorSpec((None, ncells), tf.float32)
        spec.name = 'gj_scatter'
        argspec.append(spec)
        spec = tf.SparseTensorSpec((None, ncells), tf.float32)
        spec.name = 'gj_gather'
        argspec.append(spec)

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

def main2():
    import tf2onnx
    import inspect
    print('#'*100)
    print('MAKING FUNCTION'.center(100))
    print('#'*100)
    tf_function = make_function(
        gj=True,
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

    print('#'*100)
    print('CONVERTING TO KERAS'.center(100))
    print('#'*100)
    from onnx2keras import onnx_to_keras
    keras_model = onnx_to_keras(onnx_model, [arg.name for arg in tf_function.argspec])
    print(keras_model)
    print('#'*100)

if __name__ == '__main__':
    main2()

