from .onnx_base_runner import OnnxBaseRunner

__all__ = ['OnnxCpuRunnerMt']

class OnnxCpuRunnerMt(OnnxBaseRunner):
    '''CPUExecutionProvider'''
    provider = 'CPUExecutionProvider'
    option = 'mt'
    device_type = 'cpu'

    def is_supported(self):
        return True
