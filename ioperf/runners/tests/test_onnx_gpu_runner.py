import pytest
import numpy as np
from ...bench.model_configuration import ModelConfiguration

from .. import OnnxCpuRunner
from .. import OnnxCUDARunner
from .. import OnnxTensorRTRunner

def test_cuda_single_ms_unconnected():
    nms = 1
    t =  OnnxCpuRunner()
    x =  OnnxCUDARunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=False)
    ns, o = x.run_unconnected(nms, m1.state,probe=True)

    t.setup_using_model_config(m2, gap_junctions=False)
    ns2, ot = t.run_unconnected(nms, m2.state,probe=True)
    
    if(np.allclose(ot,o) == False ):
        print("cuda")
        print(o[-1])
        print("cpu base")
        print(ot[-1]) 
        print(o.shape)
        print(ot.shape)       
        raise Exception("Sorry, outcome is not close enough to cpu baseline")

def test_cuda_single_ms_connected():
    nms = 1
    t  = OnnxCpuRunner()
    x  = OnnxCUDARunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=True)
    ns, o = x.run_with_gap_junctions(nms, m1.state, gj_src=m1.gj_src, gj_tgt=m1.gj_tgt,probe=True)

    t.setup_using_model_config(m2, gap_junctions=True)
    ns2, ot = t.run_with_gap_junctions(nms,m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)
    
    if(np.allclose(ot,o) == False ):
        print("cuda")
        print(o)
        print("cpu base")
        print(ot) 
        print(o.shape)
        print(ot.shape)       
        raise Exception("Sorry, outcome is not close enough to cpu baseline")

def test_cuda_single_ms_unconnected_with_probe():
    x = OnnxCUDARunner()
    m = ModelConfiguration.create_new(nneurons=64,seed=40)
    x.setup_using_model_config(m, gap_junctions=False)
    x.run_unconnected(1, m.state, probe=False)

def test_cuda_single_ms_connected_with_probe():
    x = OnnxCUDARunner()
    m = ModelConfiguration.create_new(nneurons=64,seed=40)
    x.setup_using_model_config(m, gap_junctions=True)
    x.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt, probe=True)

def test_tensorRT_single_ms_unconnected():
    nms = 1
    t =  OnnxCpuRunner()
    x =  OnnxTensorRTRunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=False)
    ns, o = x.run_unconnected(nms, m1.state,probe=True)

    t.setup_using_model_config(m2, gap_junctions=False)
    ns2, ot = t.run_unconnected(nms, m2.state,probe=True)
    
    if(np.allclose(ot,o) == False ):
        print("tensor rt")
        print(o[-1])
        print("cpu base")
        print(ot[-1]) 
        print(o.shape)
        print(ot.shape)       
        raise Exception("Sorry, outcome is not close enough to cpu baseline")

def test_tensorRT_single_ms_connected():
    nms = 1
    t  = OnnxCpuRunner()
    x  = OnnxTensorRTRunner()

    m1 = ModelConfiguration.create_new(nneurons=64,seed=40)
    m2 = ModelConfiguration.create_new(nneurons=64,seed=40)

    x.setup_using_model_config(m1, gap_junctions=True)
    ns, o = x.run_with_gap_junctions(nms, m1.state, gj_src=m1.gj_src, gj_tgt=m1.gj_tgt,probe=True)

    t.setup_using_model_config(m2, gap_junctions=True)
    ns2, ot = t.run_with_gap_junctions(nms,m2.state, gj_src=m2.gj_src, gj_tgt=m2.gj_tgt,probe=True)
    
    if(np.allclose(ot,o, rtol=1e-05, atol=1) == False ):
        print("tensorRT")
        print(o)
        print("cpu base")
        print(ot) 
        print(o.shape)
        print(ot.shape)       
        raise Exception("Sorry, outcome is not close enough to cpu baseline")

def test_tensorRT_single_ms_unconnected_with_probe():
    x = OnnxTensorRTRunner()
    m = ModelConfiguration.create_new(nneurons=64,seed=40)
    x.setup_using_model_config(m, gap_junctions=False)
    tt, o = x.run_unconnected(2, m.state, probe=True)
    # print(o)
    # raise NotImplementedError()

def test_tensorRT_single_ms_connected_with_probe():
    x = OnnxTensorRTRunner()
    m = ModelConfiguration.create_new(nneurons=64,seed=40)
    x.setup_using_model_config(m, gap_junctions=True)
    tt, o = x.run_with_gap_junctions(2, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt, probe=True)
    # print(o)
    # raise NotImplementedError()