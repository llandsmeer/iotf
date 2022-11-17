import numpy as np
import tensorflow as tf

def make_initial_neuron_state(
        ncells,

        # Soma State
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
