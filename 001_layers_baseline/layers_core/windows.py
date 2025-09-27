from typing import Dict, List, Tuple, Iterable


class WindowManager:
    """Rolling window manager for copy-collapse detection.

    Maintains a per-(lens, prompt_id, variant) window of the last k top-1 IDs.
    Methods mirror the inline version previously embedded in run.py.
    """

    def __init__(self, window_k: int, *, extra_window_ks: Iterable[int] | None = None):
        strict_k = int(window_k)
        extra_values = [int(k) for k in (extra_window_ks or []) if int(k) > 0]
        max_k = strict_k
        if extra_values:
            max_k = max(strict_k, max(extra_values))
        self.window_k = strict_k
        self.max_window_k = max_k
        self.soft_window_ks = sorted({strict_k, *extra_values})
        self.windows: Dict[Tuple[str, str, str], List[int]] = {}

    def append_and_trim(self, lens_type: str, prompt_id: str, variant: str, token_id: int) -> list[int]:
        key = (lens_type, prompt_id, variant)
        wl = self.windows.setdefault(key, [])
        wl.append(int(token_id))
        if len(wl) > self.max_window_k:
            wl.pop(0)
        return wl.copy()

    def get_window(self, lens_type: str, prompt_id: str, variant: str, k: int) -> List[int]:
        key = (lens_type, prompt_id, variant)
        wl = self.windows.get(key, [])
        k = max(1, int(k))
        if k >= len(wl):
            return wl.copy()
        return wl[-k:].copy()

    def reset_variant(self, prompt_id: str, variant: str) -> None:
        keys = [key for key in self.windows if key[1] == prompt_id and key[2] == variant]
        for key in keys:
            self.windows.pop(key, None)
