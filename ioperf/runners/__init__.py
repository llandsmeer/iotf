'''
Runners module. Contains both abstract and concrete
implementations. Runners are expected to implement the BaseRunner class.
'''

from .base_runner import BaseRunner
from .graphcore_runner import GraphcoreRunner
from .graphcore_streaming_runner import GraphcoreStreamingRunner
from .groqchip_runner import GroqchipRunner
from .groqchip_runner_opt1 import GroqchipRunnerOpt1
from .groqchip_runner_opt2_nocopy import GroqchipRunnerOpt2NoCopy
from .onnx_base_runner import OnnxBaseRunner
from .onnx_cpu_runner import OnnxCpuRunner
from .onnx_cpu_runner_mt import OnnxCpuRunnerMt
from .onnx_cuda_runner import OnnxCUDARunner
from .onnx_tensorrt_runner import OnnxTensorRTRunner
from .tf_base_runner import TfBaseRunner

__all__ = [
    'BaseRunner',
    'GroqchipRunner',
    'GroqchipRunnerOpt1',
    'GroqchipRunnerOpt2NoCopy',
    'GraphcoreRunner',
    'GraphcoreStreamingRunner',
    'OnnxBaseRunner',
    'OnnxCpuRunner',
    'OnnxCpuRunnerMt',
    'OnnxCUDARunner',
    'OnnxTensorRTRunner',
    'TfBaseRunner',
    'runners'
]

runners = [
    GroqchipRunner,
    GroqchipRunnerOpt2NoCopy,
    GraphcoreRunner,
    GraphcoreStreamingRunner,
    OnnxCpuRunner,
    # OnnxCpuRunnerMt,
    TfBaseRunner,
    OnnxCUDARunner,
    OnnxTensorRTRunner
]
'''
Available runners with implementation
'''

