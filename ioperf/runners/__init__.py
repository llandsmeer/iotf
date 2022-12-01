'''
Runners module. Contains both abstract and concrete
implementations. Runners are expected to implement the BaseRunner class.
'''

from .base_runner import BaseRunner
from .graphcore_runner import GraphcoreRunner
from .groqchip_runner import GroqchipRunner
from .groqchip_runner_opt1 import GroqchipRunnerOpt1
from .groqchip_runner_opt2_nocopy import GroqchipRunnerOpt2NoCopy
from .onnx_base_runner import OnnxBaseRunner
from .onnx_cpu_runner import OnnxCpuRunner
from .onnx_cpu_runner_mt import OnnxCpuRunnerMt
from .onnx_cuda_runner import OnnxCUDARunner
from .onnx_tensorrt_runner import OnnxTensorRTRunner
from .tf_base_runner import TfBaseRunner
from .tf_tpu_runner import TfTpuRunner

__all__ = [
    'BaseRunner',
    'GroqchipRunner',
    'GroqchipRunnerOpt1',
    'GroqchipRunnerOpt2NoCopy',
    'GraphcoreRunner',
    'OnnxBaseRunner',
    'OnnxCpuRunner',
    'OnnxCpuRunnerMt',
    'OnnxCUDARunner',
    'OnnxTensorRTRunner',
    'TfBaseRunner',
    'TfTpuRunner',
    'runners'
]

runners = [
    # GroqchipRunner,
    # GroqchipRunnerOpt2NoCopy,
    # GraphcoreRunner,
    # OnnxCpuRunner,
    # OnnxCpuRunnerMt,
    # TfBaseRunner,
    TfTpuRunner
    # OnnxCUDARunner,
    # OnnxTensorRTRunner
]
'''
Available runners with implementation
'''

