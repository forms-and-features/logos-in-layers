#!/usr/bin/env python3
"""Unit tests for Prism Procrustes fit.

CPU-only; constructs a W_U with a known left-singular subspace and verifies
that Q maps E_k onto U_k and is orthogonal.
"""

import _pathfix  # noqa: F401

import torch

from layers_core.prism import compute_reservoir_basis, fit_prism_Q, _left_singular_vectors_WU, orthogonality_error


def _orthonormal(d, k):
    A = torch.randn(d, k)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q[:, :k]


def test_procrustes_maps_basis():
    torch.manual_seed(123)
    d, k = 16, 6
    # Construct U_k explicitly; build W_U with left-singular vectors = U_k
    U_k = _orthonormal(d, k)
    # Set singular values (descending)
    sigma = torch.linspace(5.0, 1.0, steps=k)
    # Choose V_k arbitrary orthonormal (k x k)
    V_k = torch.linalg.qr(torch.randn(k, k), mode="reduced")[0]
    # Build W_U = U_k Σ V_k^T (so v=k); extend to (d×k)
    W_U = U_k @ torch.diag(sigma) @ V_k.T

    # Let E_k = U_k (ideal alignment)
    E_k = U_k.clone()
    Q = fit_prism_Q(W_U, E_k)
    # Check orthogonality
    assert orthogonality_error(Q) < 1e-5
    # Check that Q maps E_k close to U_k (on subspace)
    mapped = Q @ E_k
    err = torch.linalg.norm(mapped - U_k) / max(1.0, torch.linalg.norm(U_k))
    assert float(err) < 1e-5

