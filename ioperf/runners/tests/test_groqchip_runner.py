import pytest
from ...bench.model_configuration import ModelConfiguration
from .. import GroqchipRunner
from .. import GroqchipRunnerOpt2NoCopy

def test_single_ms_unconnected():
    x = GroqchipRunner()
    m = ModelConfiguration.create_new(nneurons=64)
    x.setup_using_model_config(m, gap_junctions=False)
    x.run_unconnected(1, m.state)

def test_single_ms_connected():
    x = GroqchipRunner()
    m = ModelConfiguration.create_new(nneurons=64)
    x.setup_using_model_config(m, gap_junctions=True)
    x.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt)


def test_opt2_single_ms_unconnected():
    x = GroqchipRunnerOpt2NoCopy()
    m = ModelConfiguration.create_new(nneurons=64)
    x.setup_using_model_config(m, gap_junctions=False)
    x.run_unconnected(1, m.state)

def test_opt2_single_ms_connected():
    x = GroqchipRunnerOpt2NoCopy()
    m = ModelConfiguration.create_new(nneurons=64)
    x.setup_using_model_config(m, gap_junctions=True)
    x.run_with_gap_junctions(1, m.state, gj_src=m.gj_src, gj_tgt=m.gj_tgt)
