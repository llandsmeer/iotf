import pytest
import numpy as np

from ...bench.model_configuration import ModelConfiguration
from .. import GroqchipRunner
from .. import GroqchipRunnerOpt2NoCopy
from .. import OnnxCpuRunner

def test_single_ms_unconnected():
    t = OnnxCpuRunner()
    x = GroqchipRunner()

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
    x = GroqchipRunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)
    
    x.setup_using_model_config(m1, gap_junctions=True)
    ns, o = x.run_with_gap_junctions(1, m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)

    t.setup_using_model_config(m1, gap_junctions=True)
    ns2, ot = t.run_with_gap_junctions(1, m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)
    
    if(np.allclose(ot,o) == False ):
        raise Exception("Sorry, outcome is not close enough to cpu baseline")
    

def test_opt2_single_ms_unconnected():
    nms = 1
    t =OnnxCpuRunner()
    x =  GroqchipRunnerOpt2NoCopy()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=False)
    ns, o = x.run_unconnected(nms, m1.state,probe=True)

    t.setup_using_model_config(m2, gap_junctions=False)
    ns2, ot = t.run_unconnected(nms, m2.state,probe=True)
    
    if(np.allclose(ot,o) == False ):
        print("groq chip")
        print(o[-1])
        print("cpu base")
        print(ot[-1]) 
        print(o.shape)
        print(ot.shape)       
        raise Exception("Sorry, outcome is not close enough to cpu baseline")
    

def test_opt2_single_ms_connected():
    nms = 1
    t  = OnnxCpuRunner()
    x  = GroqchipRunnerOpt2NoCopy()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=True)
    ns, o = x.run_with_gap_junctions(nms, m1.state, gj_src=m1.gj_src, gj_tgt=m1.gj_tgt,probe=True)

    t.setup_using_model_config(m2, gap_junctions=True)
    ns2, ot = t.run_with_gap_junctions(nms,m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)
    
    if(np.allclose(ot,o) == False ):
        print("groq chip")
        print(o)
        print("cpu base")
        print(ot) 
        print(o.shape)
        print(ot.shape)       
        raise Exception("Sorry, outcome is not close enough to cpu baseline")