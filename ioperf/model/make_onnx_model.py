import tempfile
import tf2onnx

from .. import model

def make_onnx_model(*args, opset=16, combine40=False, **kwargs):
    '''
    This function takes the same arguments as the make_tf_function call
    and returns the path to the ONNX model.
    Optionally, one can specify an extra `opset` argument for ONNX
    '''
    if combine40 == True:
        tf_function = model.make_tf_function_40(*args, **kwargs)
    else:
        tf_function = model.make_tf_function(*args, **kwargs)

    path = tempfile.mktemp() + '.onnx'

    onnx_model, _ = tf2onnx.convert.from_function(
            function=tf_function,
            input_signature=tf_function.argspec,
            output_path=path,
            opset=opset,
            )

    return path
