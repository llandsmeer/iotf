import sys
import pytest
import tensorflow
import numpy as np

from ...bench.model_configuration import ModelConfiguration
from .. import GraphcoreRunner
from .. import OnnxCpuRunner

class IPU_Module_Shim:
    @classmethod
    def register(cls):
        module = cls()
        return module

    def __enter__(self):
        import tensorflow.python as x
        sys.modules['ipu'] = self
        sys.modules['tensorflow.python'] = x
        x.ipu = self.ipu

    @property
    def ipu(self):
        return self

    @property
    def config(self):
        return self

    @property
    def ipu_strategy(self):
        return self

    @property
    def loops(self):
        return self

    def repeat(self, n, body, x):
        for _ in range(n):
            x = body(x)
        return x

    def __exit__(self, _e, _v, _tb):
        assert sys.modules['ipu'] is self
        assert sys.modules['tensorflow.python'].ipu is self
        del sys.modules['ipu']
        del sys.modules['tensorflow.python'].ipu

    class IPUConfig:
        def configure_ipu_system(self):
            pass

        @property
        def optimizations(self):
            return self

        @property
        def math(self):
            return self

    class IPUStrategy:
        def scope(self):
            return self

        def __enter__(self):
            pass

        def __exit__(self, _a, _b, _c):
            pass

def test_single_ms_unconnected():
    with IPU_Module_Shim.register():
        t = OnnxCpuRunner()
        x = GraphcoreRunner()

        m = ModelConfiguration.create_new(nneurons=64,seed=40)

        x.setup_using_model_config(m, gap_junctions=False)
        ns, o = x.run_unconnected(1, m.state,probe=True)
        print(o)

        t.setup_using_model_config(m, gap_junctions=False)
        ns2, ot = t.run_unconnected(1, m.state,probe=True)
        print(ot)

        if not np.allclose(ot, o):
            raise Exception("Sorry, outcome is not close enough to cpu baseline")


def test_single_ms_connected():
    with IPU_Module_Shim.register():
        t = OnnxCpuRunner()
        x = GraphcoreRunner()

        m = ModelConfiguration.create_new(nneurons=64,seed=40)

        x.setup_using_model_config(m, gap_junctions=True)
        ns, o = x.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt,probe=True)
        print(ns.shape, o.shape)

        t.setup_using_model_config(m, gap_junctions=True)
        ns2, ot = t.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt,probe=True)
        print(ns2.shape, ot.shape)

        if(np.allclose(ot,o) == False ):
            raise Exception("Sorry, outcome is not close enough to cpu baseline")
