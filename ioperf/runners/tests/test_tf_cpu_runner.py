import pytest
import numpy as np
from ...bench.model_configuration import ModelConfiguration
from .. import OnnxCpuRunner
from .. import TfBaseRunner

def test_single_ms_unconnected():
    t = OnnxCpuRunner()
    x = TfBaseRunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=False)
    ns, o = x.run_unconnected(1, m1.state,probe=True)

    t.setup_using_model_config(m2, gap_junctions=False)
    ns2, ot = t.run_unconnected(1, m2.state,probe=True)
    
    if(np.allclose(ot,o) == False ):
        raise Exception("Sorry, outcome is not close enough to cpu baseline")


def test_single_ms_connected():
    t = OnnxCpuRunner()
    x = TfBaseRunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)
    
    x.setup_using_model_config(m1, gap_junctions=True)
    ns, o = x.run_with_gap_junctions(1, m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)

    t.setup_using_model_config(m1, gap_junctions=True)
    ns2, ot = t.run_with_gap_junctions(1, m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)
    
    if(np.allclose(ot,o) == False ):
        raise Exception("Sorry, outcome is not close enough to cpu baseline")
    
