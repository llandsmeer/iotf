import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import time

from .. import model
from .base_runner import BaseRunner

__all__ = ['OnnxBaseRunner']

class OnnxBaseRunner(BaseRunner):
    '''
    Base ONNX implementation.
    Override `provider` or `make_ort_session()`
    in child classes to specify options.
    '''
    provider = '<placeholder>'

    def is_supported(self):
        return self.provider in ort.get_available_providers()

    def setup(self, *, ngj, ncells, argconfig, opset=16):
        self.onnx_path = model.make_onnx_model(ngj=ngj, ncells=ncells, argconfig=argconfig, opset=opset)
        self.ort_session = self.make_ort_session()

    def make_ort_session(self):
        sess_options = ort.SessionOptions()
        sess_options.use_deterministic_compute = True
        # GRAPH optimalizations
        #   GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
        #   GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
        #   GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
        #   GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED 
        # sess_options.optimized_model_filepath = "/tmp/io_optimized_model.onnx"  #safe the optimilized graph here!

        # ENABLE PROFILING
        # sess_options.enable_profiling = True

        # ENABLE MULTI TREAD / NODE   (intra == openMP inside a node, INTRA == multiNODE)
        # opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # ORT_PARALLEL
        # opts.inter_op_num_threads = 0
        # opts.intra_op_num_threads = 0  #Inter op num threads (used only when parallel execution is enabled) is not affected by OpenMP settings and should always be set using the ORT APIs.
        return ort.InferenceSession(self.onnx_path, sess_options, providers=[self.provider])

    def run_unconnected(self, nms, state, probe=False, **kwargs):
        args = { 'state': ort.OrtValue.ortvalue_from_numpy(state.numpy()) }
        for k, v in kwargs:
            args[k] = ort.OrtValue.ortvalue_from_numpy(v)
        args.update(kwargs)
        return self._run(nms, args, probe)

    def run_with_gap_junctions(self, nms, state, gj_src, gj_tgt, g_gj=0.05, probe=False, **kwargs):
        args = {
            'state': ort.OrtValue.ortvalue_from_numpy(state.numpy()),
            'gj_src': ort.OrtValue.ortvalue_from_numpy(gj_src.numpy()),
            'gj_tgt': ort.OrtValue.ortvalue_from_numpy(gj_tgt.numpy()),
            'g_gj': ort.OrtValue.ortvalue_from_numpy(np.array(g_gj, dtype='float32'))
        }
        args.update(kwargs)
        return self._run(nms, args, probe)

    def _run(self, nms, args, probe):
        if probe:
            trace = []
            trace.append(args['state'].numpy()[0, :])
        for _ in range(nms):
            for _ in range(40):
                outputs = self.ort_session.run(None, args)
                args['state'] = outputs[0]
            if probe:
                trace.append(args['state'][0, :])
        if probe:
            return tf.constant(args['state']), np.array(trace)
        else:
            return tf.constant(args['state'])
