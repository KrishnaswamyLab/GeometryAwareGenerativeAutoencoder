import numpy as np

# compute JS-divegence with np broadcasting.
def fast_jsd(mat):
    plogpmat = mat * np.log(mat)
    mtensor = 1/2 * (mat[:, np.newaxis, :] + mat[np.newaxis, :, :])
    mlogmtensor = mtensor * np.log(mtensor)
    return 1/2 * (plogpmat[:, np.newaxis, :] + plogpmat[np.newaxis, :, :] - 2 * mlogmtensor).sum(axis=-1)

# Test
def test_fast_jsd():
    a = np.random.rand(10,10) + 1
    jsd_mat = np.zeros((len(a), len(a)))
    for i in range(len(a)):
        for j in range(i, len(a)):
            jsd = compute_jsd(a[i], a[j])
            jsd_mat[i, j] = jsd
            jsd_mat[j, i] = jsd
    assert(np.isclose(jsd_mat, fast_jsd(a)).all())

if __name__ == '__main__':
    test_fast_jsd()