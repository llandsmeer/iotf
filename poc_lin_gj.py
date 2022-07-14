import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import collections
import numpy as np
import tensorflow as tf

def linear_gap_junctions_to_sparse_matrix(pairs, size=0):
    # tensorflow sparse array is constructed COO
    # a dictionary mapping indices (i, j) to values (v)
    # which we give as two separate lists
    diagonal = collections.defaultdict(int)
    indices = []
    values = []
    # seen is just to check we're not updating the same index
    # twice, which is not possible (but should not happend anyway
    # if we are passed a correct pairs list)
    seen = set()
    for w, i, j in pairs:
        i, j = sorted([i, j])
        # check if we're not adding a connection twice
        assert (i, j) not in seen
        seen.add((i, j))
        # if size was not given or too small, update to maximum cell index
        size = max(size, i+1, j+1)
        # connection of J onto I
        diagonal[i] -= w
        indices.append((i, j))
        values.append(w)
        # connection of I onto J
        diagonal[j] -= w
        indices.append((j, i))
        values.append(w)
    # we handle the diagonal differently because multiple connections update it
    for i, v in diagonal.items():
        indices.append((i, i))
        values.append(v)
    return tf.SparseTensor(
            indices=np.array(indices, dtype='float32'),
            values=np.array(values, dtype='float32'),
            dense_shape=(size, size)
            )

# number of cells
ncells = 4

# array of bidirectional linear gap junctions with tuples (weight, idx1, idx2)
gjs = [ (0.05, 0, 1), (0.01, 1, 2), (0.03, 2, 3) ]

# linear gap junction sparse array
w = linear_gap_junctions_to_sparse_matrix(gjs, ncells)

# random cell voltages
v = tf.convert_to_tensor(np.random.random(ncells).reshape((-1, 1)), dtype='float32')

# resulting gap junction current
actual = tf.sparse.sparse_dense_matmul(w, v)

# validation: explicit calculation of gap junction current without sparse matrix
expected = np.zeros(ncells)
for w, i, j in gjs:
    expected[i] += w * (v[j] - v[i])
    expected[j] += w * (v[i] - v[j])

# comparison
actual = actual.numpy().flatten()
print(expected)
print(actual)
assert np.allclose(actual, expected)

