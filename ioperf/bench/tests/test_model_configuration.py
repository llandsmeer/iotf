import numpy as np
import tempfile
from ..model_configuration import ModelConfiguration

def test_ncells_ngjs():
    x = ModelConfiguration.create_new(nneurons=10**3)
    assert x.ncells == 1000
    x = ModelConfiguration.create_new(nneurons=10**3)
    assert x.ngj > 1000
    x = ModelConfiguration.create_new(nneurons=4**3, rmax=2)
    assert x.ncells == 64

def test_save_load_equal():
    name = tempfile.mktemp()
    x = ModelConfiguration.create_new(nneurons=10**3)
    x.save(name)
    y = ModelConfiguration.load(name)
    assert x == y

def test_model_config_random_equal():
    '''
    numpy random.random() is the same across machines
    up to certain guarantees (eg same numpy version)
    '''
    seed_value = 1234
    np.random.seed(seed_value)
    x = ModelConfiguration.create_new(nneurons=10**3)
    y = ModelConfiguration.create_new(nneurons=10**3)
    assert x != y
    np.random.seed(seed_value)
    y = ModelConfiguration.create_new(nneurons=10**3)
    assert x == y
