import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import onnx
import tf2onnx
import tensorflow as tf

# Build Loop Body
fn = '/tmp/loop.onnx'
i = tf.keras.Input((), dtype='int64', name='i', batch_size=1)
x = tf.keras.Input((), dtype='float32', name='x', batch_size=1)
c = tf.keras.Input((), dtype='bool', name='cond_in', batch_size=1)
model = tf.keras.Model(inputs=[i, c, x], outputs=[tf.identity(c), x+1])
model, ext = tf2onnx.convert.from_keras(model, opset=16)
assert ext is None

# Build Loop Node
niter_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['niter'],
    value=onnx.helper.make_tensor(
        name='const_tensor_niter',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[40],
    )
)
cond_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['cond'],
    value=onnx.helper.make_tensor(
        name='const_tensor_cond',
        data_type=onnx.TensorProto.BOOL,
        dims=(1,),
        vals=[True],
    )
)
node = onnx.helper.make_node(
    'Loop',
    inputs=['niter', 'cond', 'loop_in'],
    outputs=['loop_out'], # scan outputs?
    body=model.graph
)
onnx.checker.check_node(niter_const_node)
onnx.checker.check_node(cond_const_node)
onnx.checker.check_node(node)

# Build Loop Graph
loop_in = onnx.helper.make_tensor_value_info(
        'loop_in', onnx.TensorProto.FLOAT, ())
loop_out = onnx.helper.make_tensor_value_info(
        'loop_out', onnx.TensorProto.FLOAT, (1,))
g = onnx.helper.make_graph(
    [niter_const_node, cond_const_node, node],
    name='loop',
    inputs=[loop_in],
    outputs=[loop_out])
onnx.checker.check_graph(g)

# Build Full Model
model = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 16)])
onnx.checker.check_model(model)

with open('/tmp/manual_reloop.onnx', 'wb') as f:
    onnx.save_model(model, f)

import os
os.system("sh c")
