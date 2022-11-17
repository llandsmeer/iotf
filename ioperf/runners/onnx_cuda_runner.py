from .onnx_base_runner import OnnxBaseRunner

__all__ = ['OnnxCUDARunner']

class OnnxCUDARunner(OnnxBaseRunner):
    provider = 'CUDAExecutionProvider'
