from . import onnx_experiment
from . import onnx_runner
from . import graphcore

__all__ = [
        'onnx_runner',
        'runners',
        'graphcore',
        'onnx_experiment'
        ]

runners = {
    'onnx': onnx_runner.main,
    'graphcore': graphcore.main
}
'''
Dictionary of runners. Keys correspond to
--runner argument of __main__.py.
'''
