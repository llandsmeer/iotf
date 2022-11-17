from .onnx_base_runner import OnnxBaseRunner

__all__ = ['OnnxCpuRunner']

class OnnxCpuRunner(OnnxBaseRunner):
    '''CPUExecutionProvider'''
    provider = 'CPUExecutionProvider'
