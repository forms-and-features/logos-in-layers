from typing import Dict, List, Tuple


class WindowManager:
    """Rolling window manager for copy-collapse detection.

    Maintains a per-(lens, prompt_id, variant) window of the last k top-1 IDs.
    Methods mirror the inline version previously embedded in run.py.
    """

    def __init__(self, window_k: int):
        self.window_k = int(window_k)
        self.windows: Dict[Tuple[str, str, str], List[int]] = {}

    def append_and_trim(self, lens_type: str, prompt_id: str, variant: str, token_id: int) -> list[int]:
        key = (lens_type, prompt_id, variant)
        wl = self.windows.setdefault(key, [])
        wl.append(int(token_id))
        if len(wl) > self.window_k:
            wl.pop(0)
        return wl.copy()

    def reset_variant(self, prompt_id: str, variant: str) -> None:
        # Reset both lens windows for this (prompt, variant)
        for lens in ("norm", "prism"):
            self.windows.pop((lens, prompt_id, variant), None)

