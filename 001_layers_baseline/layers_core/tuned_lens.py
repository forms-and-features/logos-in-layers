"""Core utilities for the Tuned Lens translator artifacts.

This module implements the affine identity-plus-low-rank translators described
in :mod:`001_layers_baseline/TUNED_LENS_PLAN.md`. A translator operates on the
architecture-correct, normalized residual stream and maps it into a basis whose
logits—decoded with the model's tied unembedding—match the model's final
distribution at the same position.

Main components
---------------

``TunedTranslator``
    Owns one ``LayerTranslator`` per post-block layer. Each layer applies an
    identity-plus-low-rank affine map, optionally in a preconditioned basis.

``Preconditioner``
    Packs whitening statistics (mean/var) and an optional orthogonal rotation.
    When present the translator works in the whitened, rotated coordinates and
    results are mapped back to the model basis before decoding.

``save_tuned_lens`` / ``load_tuned_lens``
    Persist translator weights, optional preconditioner, and metadata to disk in
    the layout mandated by the plan (``weights.pt``, ``precond.pt``,
    ``provenance.json``).

All tensors are stored in float32 for numerical stability. Callers may move the
translator to a target device/dtype after loading.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import json
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Preconditioning helpers
# -----------------------------------------------------------------------------


@dataclass
class Preconditioner:
    """Diagonal whitening with optional orthogonal rotation.

    Attributes are float32 CPU tensors; callers may move them to a specific
    device after loading. ``inv_scale`` stores :math:`1 / \sqrt{\mathrm{var}+ε}`
    to avoid repeated square roots at runtime. ``rotation`` can be ``None`` to
    indicate pure diagonal whitening.
    """

    mean: torch.Tensor
    inv_scale: torch.Tensor
    rotation: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "Preconditioner":
        return Preconditioner(
            mean=self.mean.to(device=device, dtype=torch.float32),
            inv_scale=self.inv_scale.to(device=device, dtype=torch.float32),
            rotation=None if self.rotation is None else self.rotation.to(device=device, dtype=torch.float32),
        )

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @torch.no_grad()
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply whitening (and optional rotation) to ``x``.

        Accepts a tensor whose last dimension matches ``dim``. Returns float32
        tensor located on ``x.device``.
        """

        x32 = x.to(dtype=torch.float32)
        mean = self.mean.to(device=x32.device, dtype=torch.float32)
        inv_scale = self.inv_scale.to(device=x32.device, dtype=torch.float32)
        centered = x32 - mean
        whitened = centered * inv_scale
        if self.rotation is not None:
            rot = self.rotation.to(device=x32.device, dtype=torch.float32)
            whitened = whitened @ rot.transpose(0, 1)
        return whitened

    @torch.no_grad()
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Map from preconditioned space back to the model basis."""

        x32 = x.to(dtype=torch.float32)
        if self.rotation is not None:
            rot = self.rotation.to(device=x32.device, dtype=torch.float32)
            x32 = x32 @ rot
        inv_scale = self.inv_scale.to(device=x32.device, dtype=torch.float32)
        scale = torch.where(inv_scale == 0, torch.zeros_like(inv_scale), 1.0 / inv_scale)
        mean = self.mean.to(device=x32.device, dtype=torch.float32)
        return x32 * scale + mean


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Translator blocks
# -----------------------------------------------------------------------------


class LayerTranslator(nn.Module):
    """Identity-plus-low-rank affine map for a single layer.

    ``rank`` may be zero (pure identity). Parameters are initialised to zero so
    the translator starts as an identity map. Bias ``c`` is optional but enabled
    by default.
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        *,
        bias: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(max(0, rank))
        self.bias = bool(bias)

        if self.rank > 0:
            U = torch.zeros(self.d_model, self.rank, dtype=dtype, device=device)
            V = torch.zeros(self.d_model, self.rank, dtype=dtype, device=device)
            self.U = nn.Parameter(U, requires_grad=trainable)
            self.V = nn.Parameter(V, requires_grad=trainable)
        else:
            self.register_parameter("U", None)
            self.register_parameter("V", None)

        if self.bias:
            c = torch.zeros(self.d_model, dtype=dtype, device=device)
            self.c = nn.Parameter(c, requires_grad=trainable)
        else:
            self.register_parameter("c", None)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, rank={self.rank}, bias={self.bias}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the affine translator to a (N×d) tensor."""

        if x.dim() != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"LayerTranslator expected (N,{self.d_model}); got {tuple(x.shape)}")

        out = x
        if self.rank > 0:
            U = self.U
            V = self.V
            out = out + (x @ U) @ V.transpose(0, 1)
        if self.bias and self.c is not None:
            out = out + self.c.unsqueeze(0)
        return out


class TunedTranslator(nn.Module):
    """Collection of layer translators with optional preconditioning."""

    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        rank: int,
        final_identity: bool = True,
        preconditioner: Optional[Preconditioner] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if rank < 0:
            raise ValueError("rank must be non-negative")

        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.final_identity = bool(final_identity)

        layers = []
        for idx in range(self.num_layers):
            trainable = not (self.final_identity and idx == self.num_layers - 1)
            layer_rank = 0 if (self.final_identity and idx == self.num_layers - 1) else self.rank
            layer = LayerTranslator(
                d_model=self.d_model,
                rank=layer_rank,
                bias=True,
                device=device,
                dtype=dtype,
                trainable=trainable,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # Preconditioner stored as buffers for persistence; may be None.
        if preconditioner is not None:
            if preconditioner.dim != self.d_model:
                raise ValueError("Preconditioner dimension mismatch")
            self.register_buffer("_precond_mean", preconditioner.mean.clone().detach().to(dtype=torch.float32))
            self.register_buffer("_precond_inv_scale", preconditioner.inv_scale.clone().detach().to(dtype=torch.float32))
            if preconditioner.rotation is not None:
                self.register_buffer("_precond_rotation", preconditioner.rotation.clone().detach().to(dtype=torch.float32))
            else:
                self.register_buffer("_precond_rotation", None)
        else:
            self.register_buffer("_precond_mean", None)
            self.register_buffer("_precond_inv_scale", None)
            self.register_buffer("_precond_rotation", None)

        self.to(device=device, dtype=dtype)

    # -- preconditioning --------------------------------------------------

    def has_preconditioner(self) -> bool:
        return self._precond_mean is not None and self._precond_inv_scale is not None

    def _apply_precond(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_preconditioner():
            return x.to(dtype=torch.float32)
        mean = self._precond_mean.to(device=x.device, dtype=torch.float32)
        inv_scale = self._precond_inv_scale.to(device=x.device, dtype=torch.float32)
        centered = x.to(dtype=torch.float32) - mean
        whitened = centered * inv_scale
        rot = self._precond_rotation
        if rot is not None:
            rot = rot.to(device=x.device, dtype=torch.float32)
            whitened = whitened @ rot.transpose(0, 1)
        return whitened

    def _invert_precond(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_preconditioner():
            return x.to(dtype=torch.float32)
        rot = self._precond_rotation
        x32 = x.to(dtype=torch.float32)
        if rot is not None:
            rot = rot.to(device=x.device, dtype=torch.float32)
            x32 = x32 @ rot
        inv_scale = self._precond_inv_scale.to(device=x.device, dtype=torch.float32)
        scale = torch.where(inv_scale == 0, torch.zeros_like(inv_scale), 1.0 / inv_scale)
        mean = self._precond_mean.to(device=x.device, dtype=torch.float32)
        return x32 * scale + mean

    # -- forward ----------------------------------------------------------

    def forward(self, h_norm: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply translator for ``layer_idx`` to normalized residuals.

        ``h_norm`` must have shape (N, d_model). Returns float32 tensor.
        """

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if h_norm.dim() != 2 or h_norm.shape[1] != self.d_model:
            raise ValueError(f"Expected (N,{self.d_model}) residuals; got {tuple(h_norm.shape)}")

        h_pre = self._apply_precond(h_norm)
        translated = self.layers[layer_idx](h_pre)
        return self._invert_precond(translated)

    # -- metadata ---------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        meta = {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "rank": self.rank,
            "final_identity": self.final_identity,
            "has_preconditioner": self.has_preconditioner(),
        }
        if self.has_preconditioner():
            meta["preconditioner"] = {
                "rotation": self._precond_rotation is not None,
            }
        return meta


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------


def save_tuned_lens(
    directory: str | Path,
    *,
    translator: TunedTranslator,
    provenance: Dict[str, Any],
    preconditioner_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Persist translator weights, optional preconditioner, and provenance.

    Returns a dict with paths to the saved files.
    """

    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    weights_path = dir_path / "weights.pt"
    torch.save(translator.state_dict(), weights_path)

    paths = {"weights": weights_path}

    if translator.has_preconditioner():
        # Preconditioner already included in state_dict, but write an explicit
        # snapshot for easier inspection / conversions.
        precond_path = dir_path / "precond.pt"
        payload = preconditioner_payload or {}
        if "mean" not in payload or "inv_scale" not in payload:
            payload = {
                "mean": translator._precond_mean.cpu(),
                "inv_scale": translator._precond_inv_scale.cpu(),
                "rotation": None if translator._precond_rotation is None else translator._precond_rotation.cpu(),
            }
        torch.save(payload, precond_path)
        paths["precond"] = precond_path

    provenance_path = dir_path / "provenance.json"
    _write_json(provenance_path, provenance)
    paths["provenance"] = provenance_path
    return paths


def load_tuned_lens(
    directory: str | Path,
    *,
    map_location: Optional[str | torch.device] = None,
) -> tuple[TunedTranslator, Dict[str, Any]]:
    """Load a tuned translator and provenance metadata from ``directory``."""

    dir_path = Path(directory)
    weights_path = dir_path / "weights.pt"
    prov_path = dir_path / "provenance.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing tuned lens weights at {weights_path}")
    if not prov_path.exists():
        raise FileNotFoundError(f"Missing tuned lens provenance at {prov_path}")

    provenance = _read_json(prov_path)

    meta = provenance.get("translator", {})
    required = {"num_layers", "d_model", "rank", "final_identity"}
    missing = required - set(meta)
    if missing:
        raise KeyError(f"Provenance missing translator fields: {sorted(missing)}")

    has_precond = bool(meta.get("has_preconditioner", False))
    precond_obj: Optional[Preconditioner] = None
    if has_precond:
        precond_path = dir_path / "precond.pt"
        if precond_path.exists():
            payload = torch.load(precond_path, map_location="cpu")
            mean = payload.get("mean")
            inv_scale = payload.get("inv_scale")
            rotation = payload.get("rotation")
            if mean is None or inv_scale is None:
                raise KeyError("precond.pt missing 'mean' or 'inv_scale'")
            precond_obj = Preconditioner(mean=mean.float(), inv_scale=inv_scale.float(), rotation=None if rotation is None else rotation.float())
        else:
            # Fall back to state_dict buffers after instantiation.
            precond_obj = None

    device_arg: Optional[torch.device] = None
    if isinstance(map_location, torch.device):
        device_arg = map_location
    elif isinstance(map_location, str):
        device_arg = torch.device(map_location)

    translator = TunedTranslator(
        num_layers=int(meta["num_layers"]),
        d_model=int(meta["d_model"]),
        rank=int(meta["rank"]),
        final_identity=bool(meta.get("final_identity", True)),
        preconditioner=precond_obj,
        device=device_arg,
        dtype=torch.float32,
    )

    state = torch.load(weights_path, map_location=map_location or "cpu")
    translator.load_state_dict(state)
    if device_arg is not None:
        translator.to(device_arg)
    return translator, provenance


def clip_rank(d_model: int) -> int:
    """Utility implementing the default width-scaled rank schedule."""

    base = max(1, d_model // 32)
    return int(max(64, min(base, 256)))


__all__ = [
    "Preconditioner",
    "LayerTranslator",
    "TunedTranslator",
    "save_tuned_lens",
    "load_tuned_lens",
    "clip_rank",
]
