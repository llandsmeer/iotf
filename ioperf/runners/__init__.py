'''
Runners module. Contains both abstract and concrete
implementations. Runners are expected to implement the BaseRunner class.
'''

from .base_runner import BaseRunner
from .graphcore_runner import GraphcoreRunner
from .groqchip_runner import GroqchipRunner
from .onnx_base_runner import OnnxBaseRunner
from .onnx_cpu_runner import OnnxCpuRunner
from .onnx_cuda_runner import OnnxCUDARunner
from .onnx_tensorrt_runner import OnnxTensorRTRunner

__all__ = [
    'BaseRunner',
    'GroqchipRunner',
    'GraphcoreRunner',
    'OnnxBaseRunner',
    'OnnxCpuRunner',
    'OnnxCUDARunner',
    'OnnxTensorRTRunner',
    'runners'
]

runners = [
    GroqchipRunner,
    GraphcoreRunner,
    OnnxCpuRunner,
    OnnxCUDARunner,
    OnnxTensorRTRunner
]
'''
Available runners with implementation
'''

