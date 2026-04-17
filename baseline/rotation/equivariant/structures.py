#
# structures.py - Helper utilities for HW2.
#
# Part 1: Rotation/flip utilities used by main.py (Q3.c.1-3)
# Part 2: Reynolds operator and equivariant basis computation (Q3.c.4)
#

import argparse
import numpy as np
import numpy.typing as npt


# ============================================================================
# Part 1: Rotation and flip utilities
# ============================================================================

def rotate(
    mat: npt.NDArray[np.generic], d: int,
    /,
) -> npt.NDArray[np.generic]:
    R"""
    Rotate a 2D matrix by d * 90 degrees.
    """
    #
    return np.rot90(mat, d)


def flip(
    mat: npt.NDArray[np.generic], d: int,
    /,
) -> npt.NDArray[np.generic]:
    R"""
    Flip a 2D matrix.
    """
    #
    if d == 0:
        #
        return mat
    elif d == 1:
        return np.flip(mat, 1)
    elif d == 2:
        return np.flip(mat, 0)
    elif d == 3:
        return np.flip(mat, (0, 1))
    else:
        #
        raise RuntimeError("Unsupported flipping argument.")


# ============================================================================
# Part 2: Reynolds operator and equivariant basis (Q3.c.4)
# ============================================================================


from functools import lru_cache

# lru_cache caches the results of previous calls to `get_equivariant_subspace`
@lru_cache(maxsize=None)
def get_equivariant_subspace(in_channels, k_in, k_out):
    """
    Compute rotation-equivariant basis for convolution kernels via Reynolds operator.
        W: R^{in_channels * k_in^2} -> R^{k_out^2}

    The representations are:
      - input:  rotate each channel's k_in x k_in patch (channels are trivial)
      - output: rotate the k_out x k_out output patch

    Args:
        in_channels: number of input channels
        k_in: input kernel spatial size
        k_out: output patch spatial size

    Returns:
        basis: [n_basis, dim_out, dim_in]  (dim_out = k_out^2, dim_in = in_channels*k_in^2)
    """
    dim_in = in_channels * k_in * k_in
    dim_out = k_out * k_out

    def transform_in(vec, degree):
        """Rotate every channel's k_in x k_in slice."""
        vec = np.reshape (vec,(in_channels,k_in,k_in))
        return np.rot90(vec,k=-degree, axes=(1,2)).flatten()

    def transform_out(vec, degree):
        """Rotate the k_out x k_out output patch."""

        vec = np.reshape (vec,(k_out,k_out))
        return rotate(vec,degree).flatten()

    
    # Build Reynolds operator:
    #   T_bar = (1/|G|) sum_g rho_out(g) kron rho_in(g^{-1})^T
    mats = []
    for degree in (0, 90, 180, 270):

        mat1 = np.apply_along_axis (lambda vec : transform_in(vec,degree=degree),0,np.eye(dim_in)) 
        mat2 = np.apply_along_axis (lambda vec : transform_out(vec,degree=degree),0,np.eye(dim_out))
        trans_mat = np.kron(mat2,mat1.T)
        mats.append(trans_mat)
    
    T_bar = sum(mats) / len(mats)

    # One can use np.allclose(T_bar, T_bar.T, atol=1e-6) to check if T_bar transpose is close to T_bar (may not be due to numerical errors)
    # If not, one may get small imaginary eigenvectors and values. **Just disregard imaginary part.**
    #  lambda_ are the eigenvalues
    #  V are the 1-left eigenvectors

    #using eigh as svd was very slow during hwk1 run
    lambda_, V = np.linalg.eigh(T_bar)          # lambda: [D], V: [D, D]

    
    basis_flat = V[:, lambda_ > 1 - 1e-6].T  # getting rid of all the "imaginary part"
    basis = basis_flat.reshape(-1, dim_out, dim_in).astype(np.float32)
    return basis