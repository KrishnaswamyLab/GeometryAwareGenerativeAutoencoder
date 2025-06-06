import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

class Procrustes():
    def __init__(self):
        self.is_fit = False
        self.mean = None
        self.mean1 = None
        self.norm = None
        self.norm1 = None
        self.R = None
        self.s = None
        self.disparity = None

    def fit_transform(self, data1, data2):
        r"""
        From https://github.com/scipy/scipy/blob/v1.11.4/scipy/spatial/_procrustes.py#L15-L131

        Procrustes analysis, a similarity test for two data sets.

        Each input matrix is a set of points or vectors (the rows of the matrix).
        The dimension of the space is the number of columns of each matrix. Given
        two identically sized matrices, procrustes standardizes both such that:

        - :math:`tr(AA^{T}) = 1`.

        - Both sets of points are centered around the origin.

        Procrustes ([1]_, [2]_) then applies the optimal transform to the second
        matrix (including scaling/dilation, rotations, and reflections) to minimize
        :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
        pointwise differences between the two input datasets.

        This function was not designed to handle datasets with different numbers of
        datapoints (rows).  If two data sets have different dimensionality
        (different number of columns), simply add columns of zeros to the smaller
        of the two.

        Parameters
        ----------
        data1 : array_like
            Matrix, n rows represent points in k (columns) space `data1` is the
            reference data, after it is standardised, the data from `data2` will be
            transformed to fit the pattern in `data1` (must have >1 unique points).
        data2 : array_like
            n rows of data in k space to be fit to `data1`.  Must be the  same
            shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

        Returns
        -------
        mtx1 : array_like
            A standardized version of `data1`.
        mtx2 : array_like
            The orientation of `data2` that best fits `data1`. Centered, but not
            necessarily :math:`tr(AA^{T}) = 1`.
        disparity : float
            :math:`M^{2}` as defined above.

        Raises
        ------
        ValueError
            If the input arrays are not two-dimensional.
            If the shape of the input arrays is different.
            If the input arrays have zero columns or zero rows.

        See Also
        --------
        scipy.linalg.orthogonal_procrustes
        scipy.spatial.distance.directed_hausdorff : Another similarity test
        for two data sets

        Notes
        -----
        - The disparity should not depend on the order of the input matrices, but
        the output matrices will, as only the first output matrix is guaranteed
        to be scaled such that :math:`tr(AA^{T}) = 1`.

        - Duplicate data points are generally ok, duplicating a data point will
        increase its effect on the procrustes fit.

        - The disparity scales as the number of points per input matrix.

        References
        ----------
        .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
        .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.spatial import procrustes

        The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
        ``a`` here:

        >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
        >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
        >>> mtx1, mtx2, disparity = procrustes(a, b)
        >>> round(disparity)
        0.0

        """
        mtx1 = np.array(data1, dtype=np.double, copy=True)
        mtx2 = np.array(data2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        self.mean1 = np.mean(mtx1, 0)
        self.mean = np.mean(mtx2, 0)
        mtx1 -= self.mean1
        mtx2 -= self.mean

        self.norm1 = np.linalg.norm(mtx1)
        self.norm = np.linalg.norm(mtx2)

        if self.norm1 == 0 or self.norm == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= self.norm1
        mtx2 /= self.norm

        # transform mtx2 to minimize disparity
        R, s = orthogonal_procrustes(mtx1, mtx2)
        mtx2 = np.dot(mtx2, R.T) * s
        # measure the dissimilarity between the two datasets
        disparity = np.sum(np.square(mtx1 - mtx2))
        
        self.R = R
        self.s = s
        self.is_fit = True
        self.disparity = disparity
        mtx1 = self.norm1 * mtx1 + self.mean1
        mtx2 = self.norm1 * mtx2 + self.mean1
        return mtx1, mtx2, disparity

    def transform(self, data2):
        assert self.is_fit
        mtx2 = np.array(data2, dtype=np.double, copy=True)
        mtx2 -= self.mean
        mtx2 /= self.norm
        mtx2 = np.dot(mtx2, self.R.T) * self.s
        mtx2 = self.norm1 * mtx2 + self.mean1
        return mtx2
    
    def inverse_transform(self, datat):
        assert self.is_fit
        mtxt = np.array(datat, dtype=np.double, copy=True)
        mtxt = (mtxt - self.mean1) / self.norm1
        mtx2 = self.norm * mtxt @ self.R / self.s + self.mean
        return mtx2

    def fit(self, data1, data2):
        self.fit_transform(data1, data2)

def proc_transf_torch(proc, device, dtype): # temp fix for taking jacobian.
    m = torch.tensor(proc.mean, dtype=dtype, device=device)
    m1 = torch.tensor(proc.mean1, dtype=dtype, device=device)
    n1 = torch.tensor(proc.norm1, dtype=dtype, device=device)
    n = torch.tensor(proc.norm, dtype=dtype, device=device)
    R = torch.tensor(proc.R, dtype=dtype, device=device)
    s = torch.tensor(proc.s, dtype=dtype, device=device)
    def transf(x):
        return ((x - m) / n @ R.T * s) * n1 + m1
    return transf
