from .onnx_base_runner import OnnxBaseRunner

__all__ = ['OnnxTensorRTRunner']

class OnnxTensorRTRunner(OnnxBaseRunner):
    '''TensorrtExecutionProvider'''
    provider = 'TensorrtExecutionProvider'
