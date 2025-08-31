"""Logit Prism (shared decoder) core utilities.

This module provides helpers to:
- Maintain running whitening moments (mean/var) over residual vectors.
- Build a low-rank basis from a reservoir sample (PCA/low-rank SVD).
- Fit an orthogonal map Q that aligns the whitened residual subspace to the
  left singular subspace of the model's unembedding (Procrustes).
- Save/load artifacts to disk and apply whitening at decode time.

Notes
- We avoid storing a separate d×V decoder; at runtime we reuse the model's W_U
  and (optional) b_U. Prism only contributes whitening stats and an orthogonal Q.
- All numerics run in float32 for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import json
import math
import torch


@dataclass
class WhitenStats:
    mean: torch.Tensor  # (d,)
    var: torch.Tensor   # (d,)
    eps: float = 1e-8


class RunningMoments:
    """Running per-feature mean/variance using a batched Welford update.

    Maintains numerical stability in float32; accepts 2D batches (N×d).
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = int(dim)
        self.eps = float(eps)
        self.n = 0
        self.mean = torch.zeros(self.dim, dtype=torch.float32)
        self.M2 = torch.zeros(self.dim, dtype=torch.float32)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update with a batch `x` of shape (N, d)."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected (N,{self.dim}) batch, got {tuple(x.shape)}")
        x32 = x.to(dtype=torch.float32)
        batch_n = x32.shape[0]
        if batch_n == 0:
            return
        batch_mean = x32.mean(dim=0)
        batch_M2 = ((x32 - batch_mean) ** 2).sum(dim=0)

        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = batch_n
            return

        # Pooled update
        delta = batch_mean - self.mean
        total_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / total_n)
        self.M2 = self.M2 + batch_M2 + (delta ** 2) * (self.n * batch_n / total_n)
        self.n = total_n

    @torch.no_grad()
    def finalize(self) -> WhitenStats:
        if self.n <= 0:
            raise RuntimeError("No data seen; cannot finalize running moments")
        var = (self.M2 / max(self.n, 1)).clamp_min(self.eps)
        return WhitenStats(mean=self.mean.clone(), var=var.clone(), eps=self.eps)


@torch.no_grad()
def whiten_apply(x: torch.Tensor, stats: WhitenStats) -> torch.Tensor:
    """Apply diagonal whitening: (x − μ) / sqrt(var + eps).

    Accepts x with last dimension d; broadcasts mean/var over leading dims.
    Returns float32 tensor.
    """
    x32 = x.to(dtype=torch.float32)
    denom = torch.sqrt(stats.var.to(x32.device, dtype=torch.float32) + float(stats.eps))
    return (x32 - stats.mean.to(x32.device, dtype=torch.float32)) / denom


@torch.no_grad()
def compute_reservoir_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """Return an orthonormal basis E_k ∈ R^{d×k} from sample rows X (N×d).

    Uses `torch.pca_lowrank` when available; falls back to SVD on centered X.
    """
    if X.dim() != 2:
        raise ValueError("X must be 2D (N×d)")
    N, d = X.shape
    if N < 2 or k <= 0:
        raise ValueError("Insufficient samples or invalid k")
    k = min(k, d)

    Xc = X.to(dtype=torch.float32) - X.mean(dim=0, keepdim=True)

    try:
        q = min(k + 8, min(N, d) - 1) if min(N, d) > 1 else k
        U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
        E_k = V[:, :k]
    except Exception:
        # Fallback: SVD on Xc
        # Xc = U Σ Vᵀ → columns of V are principal axes (d×d)
        V = torch.linalg.svd(Xc, full_matrices=False).Vh.transpose(-1, -2)
        E_k = V[:, :k]

    # Ensure orthonormal columns via a QR polish
    E_k, _ = torch.linalg.qr(E_k, mode="reduced")
    return E_k[:, :k].contiguous()


@torch.no_grad()
def _left_singular_vectors_WU(W_U: torch.Tensor, k: int) -> torch.Tensor:
    """Compute top-k left singular vectors of W_U via eigendecomposition of W_U W_Uᵀ.

    Returns U_k ∈ R^{d×k}. Operates on CPU in float32 to avoid device-specific
    linalg gaps (e.g., torch.linalg.eigh on MPS).
    """
    d, v = W_U.shape
    k = min(k, d)
    # Move to CPU float32 for stable, portable linear algebra
    W_cpu = W_U.detach().to(device="cpu", dtype=torch.float32)
    C = W_cpu @ W_cpu.transpose(0, 1)  # (d×d) PSD
    # Symmetric PSD → use eigh; ascending eigenvalues, take largest k
    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)[:k]
    U_k = evecs[:, idx]
    # Orthonormalize defensively
    U_k, _ = torch.linalg.qr(U_k, mode="reduced")
    return U_k[:, :k].contiguous()


@torch.no_grad()
def fit_prism_Q(W_U: torch.Tensor, E_k: torch.Tensor) -> torch.Tensor:
    """Fit orthogonal Q (d×d) minimizing ||Q E_k − U_k||_F.

    - W_U: model unembedding matrix (d×vocab)
    - E_k: residual subspace basis (d×k), orthonormal columns
    Solution: SVD of U_k E_kᵀ → U Σ Vᵀ, then Q = U Vᵀ.
    """
    if E_k.dim() != 2:
        raise ValueError("E_k must be 2D (d×k)")
    d, k = E_k.shape
    if W_U.dim() != 2 or W_U.shape[0] != d:
        raise ValueError("W_U shape must be (d×v) matching E_k rows")

    U_k = _left_singular_vectors_WU(W_U, k)
    M = U_k @ E_k.transpose(0, 1)  # (d×d) rank ≤ k
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vh
    # Final polish to enforce orthogonality (Q ← UVᵀ of its own SVD)
    Uq, _, Vhq = torch.linalg.svd(Q, full_matrices=False)
    Q = Uq @ Vhq
    return Q.to(dtype=torch.float32).contiguous()


@torch.no_grad()
def orthogonality_error(Q: torch.Tensor) -> float:
    I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
    err = torch.linalg.norm(Q.transpose(0, 1) @ Q - I).item()
    return float(err)


def _torch_save(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.cpu(), str(path))


def _json_write(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_prism_artifacts(art_dir: str | Path, *, stats: WhitenStats, Q: torch.Tensor, provenance: Dict) -> Tuple[Path, Path, Path]:
    """Save whitening stats, Q, and provenance JSON under art_dir.

    Returns paths: (whiten.pt, Q_prism.pt, provenance.json)
    """
    base = Path(art_dir)
    w_path = base / "whiten.pt"
    q_path = base / "Q_prism.pt"
    p_path = base / "provenance.json"

    # Pack whitening stats into a CPU fp32 dict for torch.save
    whit = {
        "mean": stats.mean.detach().cpu().to(dtype=torch.float32),
        "var": stats.var.detach().cpu().to(dtype=torch.float32),
        "eps": float(stats.eps),
    }
    _torch_save(Q.to(dtype=torch.float32).detach().cpu(), q_path)
    torch.save(whit, str(w_path))
    _json_write(provenance, p_path)
    return w_path, q_path, p_path


def load_prism_artifacts(art_dir: str | Path) -> Tuple[WhitenStats, torch.Tensor, Dict]:
    """Load whitening stats, Q, and provenance from art_dir."""
    base = Path(art_dir)
    w_path = base / "whiten.pt"
    q_path = base / "Q_prism.pt"
    p_path = base / "provenance.json"

    if not (w_path.exists() and q_path.exists() and p_path.exists()):
        raise FileNotFoundError(f"Missing prism artifacts in {base}")

    whit = torch.load(str(w_path), map_location="cpu")
    Q = torch.load(str(q_path), map_location="cpu")
    prov = json.loads(p_path.read_text(encoding="utf-8"))

    stats = WhitenStats(mean=whit["mean"].to(dtype=torch.float32), var=whit["var"].to(dtype=torch.float32), eps=float(whit.get("eps", 1e-8)))
    return stats, Q.to(dtype=torch.float32), prov
