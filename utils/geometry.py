# SPDX-License-Identifier: Apache-2.0
import numpy as np
from typing import Tuple

def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray=None) -> Tuple[float, np.ndarray]:
    """Computes the SE3 transform from the src to dst points with the Umeyama method.

    S. Umeyama "Least-Squared Estimation of Transformation Parameters Between Two Point Patterns", TPAMI 1991.
    
    An importance weight can be assigned to each point, which can be used to implement robust alignment with IRLS.

    Args:
        src: The source points with shape (N,D).
        dst: The destination points corresponding to the src array with shape (N,D).
        weights: Optional array with scalar weights to make points more or less important. The shape is (N,)

    Returns:
        Returns a tuple (c, Rt) with the scaling factor and the rigid transformation as 4x4 matrix.
        A source point x as a homogeneous column vector can be transformed to the destination minimizing the least squares distances with::
            y = Rt @ (c*x)

        To transform src to dst with numpy use::
            # with inhomogeneous coordinates
            dst2 = (c*src) @ Rt[:3,:3].T + Rt[:3,3]

            # with homogeneous coordinates
            csrc_hom = np.concatenate((c*src, np.ones_like(src[:,:1])), axis=-1)
            dst2 = csrc_hom @ Rt[:3,:].T
    """
    assert src.ndim == 2
    assert src.shape == dst.shape
    dtype = src.dtype
    dim = src.shape[1]

    if weights is None:
        weights = np.ones((src.shape[0],), dtype=dtype)
    assert weights.shape == src.shape[:1]

    w = weights[...,None]
    weights_sum = weights.sum()

    # Compute means and variances as in equations (34)-(38)
    src_mean = np.sum(src * w, axis=0, keepdims=True)/weights_sum
    dst_mean = np.sum(dst * w, axis=0, keepdims=True)/weights_sum
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    src_var = ((src_demean**2).sum(axis=1)*weights).sum()/weights_sum

    sigma = ((dst_demean).T @ (src_demean*w)) / weights_sum

    svd = np.linalg.svd(sigma)
    Rt = np.eye(dim+1, dtype=dtype)

    # Compute rotation without reflection (39),(40),(43)
    S = np.ones((dim,), dtype=dtype)
    S[-1] = -1 if np.linalg.det(sigma) < 0 else 1

    if np.linalg.matrix_rank(sigma) == dim-1:
        if np.linalg.det(svd[0]) * np.linalg.det(svd[2]) > 0:
            Rt[:-1,:-1] = svd[0] @ svd[2]
        else:
            S2 = np.ones((dim,), dtype=dtype)
            S2[-1] = -1
            Rt[:-1,:-1] = svd[0] @ np.diag(S2) @ svd[2]
    else:
        Rt[:-1,:-1] = svd[0] @ np.diag(S) @ svd[2]

    # scale (42)
    c = (svd[1]*S).sum()/src_var

    # translation (41)
    Rt[:-1,-1] = dst_mean
    Rt[:-1,dim:] -= c*Rt[:-1,:-1]@(src_mean.T)

    return c, Rt

