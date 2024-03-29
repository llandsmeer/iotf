import tensorflow as tf

NUM_STATE_VARS = 14

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
        g_gj            = 0.05,
        unroll_gj       = 0
        ):
    '''
    Raw python implementation of the IO model using tensorflow math functions.
    Don't call this function directly, instead use make_tf_function() to
    optimize out unused parameters and compile to a tf.function for later use.
    '''

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

    if unroll_gj == 0 and (gj_src is not None and gj_tgt is not None):
        vdiff = tf.gather(V_dend, gj_src) - tf.gather(V_dend, gj_tgt)
        cx36_current_per_gj = (0.2 + 0.8 * tf.exp(-vdiff*vdiff / 100)) * vdiff * g_gj
        I_gapp = tf.tensor_scatter_nd_add(tf.zeros_like(V_dend), tf.reshape(gj_tgt, (-1, 1)),
            cx36_current_per_gj)
    elif gj_src is not None and gj_tgt is not None:
        I_gapp = 0
        for i in range(0, gj_src.shape[0], unroll_gj):
            end = min(i + unroll_gj, gj_src.shape[0])
            vdiff = tf.gather(V_dend, gj_src[i:end]) - tf.gather(V_dend, gj_tgt[i:end])
            cx36_current_per_gj = (0.2 + 0.8 * tf.exp(-vdiff*vdiff / 100)) * vdiff * g_gj
            I_gapp += tf.tensor_scatter_nd_add(tf.zeros_like(V_dend), tf.reshape(gj_tgt[i:end], (-1, 1)), cx36_current_per_gj)
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
