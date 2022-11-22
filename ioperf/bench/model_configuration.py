import numpy as np
import tensorflow as tf

from .. import model

__all__ = ['ModelConfiguration']

class ModelConfiguration:
    def __init__(self, *, state, gj_src, gj_tgt):
        assert isinstance(state, tf.Tensor)
        assert isinstance(gj_src, tf.Tensor)
        assert isinstance(gj_tgt, tf.Tensor)
        self.state = state
        self.gj_src = gj_src
        self.gj_tgt = gj_tgt

    @property
    def ncells(self):
        return int(self.state.shape[1])

    @property
    def ngj(self):
        return int(self.gj_src.shape[0])

    @classmethod
    def create_new(cls, nneurons, rmax=4, seed=None):
        if seed is not None:
            np.random.seed(seed) # this is ugly
        state = model.make_initial_neuron_state(nneurons, dtype=tf.float32, V_axon=None, V_dend=None, V_soma=None)
        gj_src, gj_tgt = model.sample_connections_3d(nneurons, rmax=rmax)
        return cls(state=state, gj_src=gj_src, gj_tgt=gj_tgt)

    def save(self, filename):
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez_compressed(
                filename,
                gj_src=self.gj_src.numpy(),
                gj_tgt=self.gj_tgt.numpy(),
                state=self.state.numpy()
                )

    @classmethod
    def load(cls, filename):
        if not filename.endswith('.npz'):
            filename += '.npz'
        f = np.load(filename)
        return cls(
                state=tf.constant(f['state']),
                gj_src=tf.constant(f['gj_src']),
                gj_tgt=tf.constant(f['gj_tgt'])
                )

    def __eq__(self, other):
        return np.allclose(self.state, other.state) and \
               np.all(self.gj_src == other.gj_src) and \
               np.all(self.gj_tgt == other.gj_tgt)

