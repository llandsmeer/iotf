import io
import tensorflow as tf

from .timestep import timestep

NUM_STATE_VARS = 14

def make_tf_function(*, ngj=0, ncells=None, argconfig=()):
    '''
    Build an optimized timestep tensorflow function.

    `ngj`: number of gap junctions (either 0 or len(output of sample_connections_3d)

    When `ngj` is non-zero, the resulting function expects an extra set of parameters

    `gj_src`, `gj_tgt` and `g_gj`

    `ncells`: number of neurons

    `argconfig`: a dictionary of argument configuration.

    keys correspond to model parameters, and can be chosen from

    `g_int`, `p1`, `p2`, `g_CaL`, `g_h`, `g_K_Ca`,
    `g_ld`, `g_la`, `g_ls`, `g_Na_s`, `g_Kdr_s`, `g_K_s`,
    `g_CaH`, `g_Na_a`, `g_K_a`, `V_Na`, `V_K`, `V_Ca`,
    `V_h`, `V_l`, `I_app`, `delta`

    values can be

     - `'CONSTANT'`: when calling the resulting function you provide
       a single number which will be constant parameter across all cells
     - `'VARY'`: when calling the resulting function you provide an array
       of length `ncells` with the parameter values for each cell
     - a numerical value (float or int): a single constant value to
       be compiled into the model (hopefully optimized out)

    The resulting function will always require the `state` argument,
    initially generated by the make_initial_neuron_state() function.
    '''
    MAKE_FUNCTION_TEMPLATE = '@tf.function\ndef wrapper({function_args}): return timestep({call_args})'
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
