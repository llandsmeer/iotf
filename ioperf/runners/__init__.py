from . import onnx_experiment
from . import onnx_runner
from . import graphcore

runners = {
    'onnx': onnx_runner.main,
    'graphcore': graphcore.main
}
