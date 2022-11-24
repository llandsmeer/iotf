import numpy as np
import tensorflow as tf

def sample_connections_3d(
        nneurons,
        nconnections=10,
        rmax=2,
        connection_probability=lambda r: np.exp(-(r/4)**2),
        normalize_by_dr=True
        ):
    '''
    (Gaussian) connection sampling in a cube. `nneurons` must thus
    be a power of 3 integer. `nconnections` is  amount of connections
    per neuron. Values in the range of 10 are biologically plausible.
    `rmax` determines the maximum radial distance considered.
    The `connection_probability` is a function that determines the
    PDF, it will be normalized according to `rmax` and uneven radial point
    distribution in a 3d lattice.
    '''
    assert int(round(nneurons**(1/3)))**3 == nneurons
    # we sample half the connections for each neuron
    assert nconnections % 2 == 0
    # we assume a cubic (4d toroid) brain
    nside = int(np.ceil(nneurons**(1/3)))
    if rmax > nside / 2: rmax = nside // 2
    # we set up a connection probability kernel around each neuron
    dx, dy, dz = np.mgrid[-rmax:rmax+1, -rmax:rmax+1, -rmax:rmax+1]
    dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    # we only sample backwards, as the forward connections
    # are part of the kernel of other neurons
    sample_backwards = \
            ((dz < 0)) | \
            ((dz == 0) &( dy < 0)) | \
            ((dz == 0) & (dy == 0) & (dx < 0))
    m = (r != 0) & sample_backwards & (r < rmax)
    dx, dy, dz, r = dx[m], dy[m], dz[m], r[m]
    P = connection_probability(r)

    # next, there is a ~r^2 increase in point density per r,
    # and very non uniform distribution of those due to
    # the integer grid. let's remove that bias
    ro, r_uniq_idx = np.unique(r, return_inverse=True)
    r_idx_freq = np.bincount(r_uniq_idx)
    r_freq = r_idx_freq[r_uniq_idx]
    P = P / r_freq
    if normalize_by_dr:
        dr = 0.5*np.diff(ro, append=rmax)[r_uniq_idx] + 0.5*np.diff(ro, prepend=0)[r_uniq_idx]
        P = P * dr
    # P must sum up to 1
    P = P / P.sum()

    # a connection connects two neurons
    final_connection_count = nneurons * nconnections // 2

    # instead of sampling using the P array,
    # we sample for each value of the P array,
    # which is much more memory efficient
    counts = (P * final_connection_count + .5).astype(int)
    counts[-1] =  max(0, final_connection_count - counts[:-1].sum())
    assert (counts < nneurons).all()
    conn_idx = []
    for draw in range(len(P)):
        if counts[draw] == 0:
            continue
        if counts[draw] == 1:
            draw_idx = np.array([np.random.randint(nneurons)])
        else:
            draw_idx = np.random.choice(nneurons, counts[draw], replace=False)
        conn_idx.append(draw + len(P) * draw_idx)
    conn_idx = np.concatenate(conn_idx)

    # now we calculate the neuron indices back from the P kernel
    neuron_id1 = conn_idx // len(P)
    x = ( neuron_id1 %  nside).astype('int32')
    y = ((neuron_id1 // nside) % nside).astype('int32')
    z = ((neuron_id1 // (nside*nside)) % nside).astype('int32')

    di = conn_idx % len(P)

    neuron_id2 = ( \
        (x + dx[di]) % nside + \
        (y + dy[di]) % nside * nside + \
        (z + dz[di]) % nside * nside * nside
        ).astype(int)

    # and generate the final index arrays
    # needed for gj calculation
    tgt_idx = np.concatenate([neuron_id1, neuron_id2])
    src_idx = np.concatenate([neuron_id2, neuron_id1])

    return tf.constant(src_idx, dtype='int32'), \
           tf.constant(tgt_idx, dtype='int32')
