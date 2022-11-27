import pytest
import numpy as np
from ...bench.model_configuration import ModelConfiguration
from .. import OnnxCpuRunner

def test_single_ms_connected():
    t = OnnxCpuRunner()
    x = OnnxCpuRunner()

    m = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m, gap_junctions=True)
    ns, o = x.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt,probe=True)
    print(ns.shape, o.shape)

    t.setup_using_model_config(m, gap_junctions=True, unroll_gj=10)
    ns2, ot = t.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt,probe=True)
    print(ns2.shape, ot.shape)

    if(np.allclose(ot,o) == False ):
        raise Exception("Sorry, outcome is not close enough to cpu baseline")

def test_single_ms_connected_with_probe():
    x = OnnxCpuRunner()
    m = ModelConfiguration.create_new(nneurons=64, seed = 40)
    x.setup_using_model_config(m, gap_junctions=True, unroll_gj=10)
    x.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt, probe=True)
