### DEFINE ###

import onnx.parser

g = '''
<
  ir_version: 7,
  opset_import: [ "" : 13 ]
>
agraph (int64[1] Loops, float[1] A, float[1] B) => (float[1] C)
{
    C = Loop(Loops) < body = bodygraph () => () {
    }>
}
'''

model = onnx.parser.parse_model(g)

onnx.checker.check_model(model)

### RUN ###

import tensorflow as tf
from onnx2keras import onnx_to_keras

f = onnx_to_keras(model, ['Loops', 'A', 'B'])

Loops = tf.constant(2)
A = tf.constant(1)
B = tf.constant(1)
C = f((Loops, A, B))
print(C)
