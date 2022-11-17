'''
Runners module. Contains both abstract and concrete
implementations

'''

from .base_runner import BaseRunner
from .graphcore_runner import GraphcoreRunner
from .onnx_base_runner import OnnxBaseRunner
from .onnx_cpu_runner import OnnxCpuRunner
from .onnx_cuda_runner import OnnxCUDARunner
from .onnx_tensorrt_runner import OnnxTensorRTRunner

__all__ = [
    'BaseRunner',
    'GraphcoreRunner',
    'OnnxBaseRunner',
    'OnnxCpuRunner',
    'OnnxCUDARunner',
    'OnnxTensorRTRunner',
    'runners'
]

runners = [
    GraphcoreRunner,
    OnnxCpuRunner,
    OnnxCUDARunner,
    OnnxTensorRTRunner
]
'''
Available runners with implementation
'''

