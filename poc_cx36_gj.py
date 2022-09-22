import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import collections
import numpy as np
import tensorflow as tf

def cx36_gap_junctions_to_sparse_matrix(pairs, ncells=0):
    # tensorflow sparse array is constructed COO
    # a dictionary mapping indices (i, j) to values (v)
    # which we give as two separate lists
    #
    # scatter is maps the voltage array to an voltage difference per gap junction array
    scatter_indices = []
    scatter_values = []
    # gather maps the conductances, derived from the voltage differences to currents
    gather_indices = []
    gather_values = []
    for idx, (w, i, j) in enumerate(pairs):
        i, j = sorted([i, j])
        # if ncells was not given or too small, update to maximum cell index
        ncells = max(ncells, i+1, j+1)
        # bidirectional connections
        for gj_id, self, other in [2*idx, i, j], [2*idx+1, j, i]:
            # scatter: vdiff[gj_id] = v[other] - v[self]
            scatter_indices.extend([(gj_id, self), (gj_id, other)])
            scatter_values.extend([1, -1])
            # gather: i[self] = sum_{gj_ids(self)} w * f(vdiff[gj_id])
            gather_indices.append((self, gj_id))
            gather_values.append(w)
    scatter = tf.SparseTensor(
            indices=np.array(scatter_indices, dtype='float32'),
            values=np.array(scatter_values, dtype='float32'),
            dense_shape=(len(pairs)*2, ncells)
            )
    gather = tf.SparseTensor(
            indices=np.array(gather_indices, dtype='float32'),
            values=np.array(gather_values, dtype='float32'),
            dense_shape=(ncells, len(pairs)*2)
            )
    return scatter, gather

# number of cells
ncells = 4

# array of bidirectional cx36 gap junctions with tuples (weight, idx1, idx2)
gjs = [ (0.05, 0, 1), (0.01, 1, 2), (0.03, 2, 3) ]

# cx36 gap junction sparse array
scatter, gather = cx36_gap_junctions_to_sparse_matrix(gjs, ncells)

# random cell voltages
v = tf.convert_to_tensor(np.random.random(ncells), dtype='float32')

# resulting gap junction current
vdiff = tf.sparse.sparse_dense_matmul(scatter, v[..., None])#tf.reshape(v, (-1, 1)))
cx36 = 0.2 + 0.8 * tf.exp(-vdiff*vdiff / 100)
current = tf.sparse.sparse_dense_matmul(gather, cx36)[..., 0]

# validation: explicit calculation of gap junction current without sparse matrix
expected = np.zeros(ncells)
for w, i, j in gjs:
    for self, other in [i, j], [j, i]:
        vdiff = v[other] - v[self]
        expected[self] += w * (0.2 + 0.8*np.exp(-vdiff**2/100))

# comparison
actual = current.numpy()
print(expected)
print(actual)
assert np.allclose(actual, expected)

