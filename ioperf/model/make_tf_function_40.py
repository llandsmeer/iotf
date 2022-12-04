import io
import tensorflow as tf

from .. import model

STEP40_TEMPLATE = '''
@tf.function(jit_compile=True)
def f(state {args}):
    for _ in range(40):
        state = tf_function(state {kwargs})['state_next']
    return {{"state_next": state}}
'''

NUM_STATE_VARS = 14

def make_tf_function_40(*, ngj=0, ncells=None, argconfig=(), unroll_gj=0):
    tf_function = model.make_tf_function(ngj=ngj, ncells=ncells, argconfig=argconfig)
    args = list(argconfig.keys())
    if ngj != 0:
        args = ['gj_src', 'gj_tgt', 'g_gj'] + args
    prefix = ', ' if args else ''
    src = STEP40_TEMPLATE.format(
            args=prefix + ', '.join(args),
            kwargs=prefix + ', '.join(f'{k}={k}' for k in args)
            )
    env = dict(
        tf=tf,
        tf_function=tf_function
    )
    exec(src, env)
    wrap = env['f']
    wrap.argspec = tf_function.argspec
    return wrap
