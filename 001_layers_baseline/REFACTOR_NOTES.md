# Refactor Plan for 001_layers_baseline/run.py

Goal: reduce `run.py` size and complexity without changing behavior. We will extract small, well‑scoped units with minimal coupling first, add tests when prudent, and verify via the existing CPU test suite. Each move should be reversible and low risk.

## Principles

- Preserve behavior and outputs byte‑for‑byte where feasible.
- Prefer extracting leaf utilities/classes first; avoid deep orchestration changes early.
- Keep interfaces explicit; pass dependencies instead of using module globals/closures.
- Add or reuse unit tests to guard semantics before touching larger flows.

## Quick Inventory of Extraction Candidates (ranked by safety/impact)

1) WindowManager (very low risk, small surface)
- What: the small rolling window helper class for copy‑collapse detection inside `run.py`.
- Why: self‑contained logic; used in three passes (orig, no_filler, control); easy to unit test; reduces duplication.
- Target: `layers_core/windows.py` (new) and export via `layers_core/__init__.py`.

2) Head transforms detection (low risk)
- What: `_detect_head_transforms()` local helper that probes `model`/`cfg` for scale/softcap.
- Why: purely introspective, no side effects; makes `run.py` slimmer.
- Target: `layers_core/head_transforms.py` (new) with a single `detect_head_transforms(model) -> tuple[Optional[float], Optional[float]]`.

3) Unembedding helper + fp32 shadow selection (low–medium risk)
- What: `_unembed_mm` closure + block that promotes analysis unembedding weights to fp32 when needed.
- Why: reused in many places; centralizes dtype/device handling; reduces clutter in `run.py`.
- Target: `layers_core/unembed.py` (new):
  - `prepare_unembed_weights(W_U, b_U, force_fp32: bool) -> (W, b, dtype_str)`
  - `unembed_mm(X, W, b, cache=None) -> logits`

4) Record construction (medium risk)
- What: `print_summary()` record builder and JSON appends.
- Why: consolidate record shaping into pure functions returning row dictionaries; easier to test.
- Target: `layers_core/records.py` (new): `make_record(...) -> dict` and `make_pure_record(...) -> dict`.

5) Pure next‑token emission (medium–higher risk)
- What: `emit_pure_next_token_record(...)` (currently a large nested helper).
- Why: concentrates metrics, flags, and optional raw‑lens sample; but has many inputs and call‑site context.
- Target: later step after (1–4) so we can pass smaller, typed inputs and reuse helpers. Keep signature explicit; return the record and a small summary for `collected_pure_records`.

6) Prism sidecar emitters (higher risk)
- What: sidecar per‑position and pure‑next‑token writing (duplicated across passes).
- Why: sizeable block; already has supporting `prism.py` utils and tests; can be isolated once (1–5) are stable.
- Target: `layers_core/prism_sidecar.py` (new) with mirror interfaces to baseline emitters.

7) CLI/launcher separation (low–medium risk)
- What: `parse_cli()` + `main()` orchestration/rotation/launching.
- Why: trims the file; reduces import‑time side effects by isolating the launcher.
- Target: `001_layers_baseline/launcher.py` (later); keep `run.py` as the worker.

## Proposed First Move (Step 1): Extract WindowManager

Rationale
- Small, self‑contained class handling a rolling window keyed by `(lens_type, prompt_id, prompt_variant)`.
- Used consistently in the three analysis passes; interface is stable and easy to test.
- No third‑party deps; only `list[int]` and dict operations.

Current shape (in `run.py`)
```python
class WindowManager:
    def __init__(self, window_k: int):
        self.window_k = window_k
        self.windows: dict[tuple[str, str, str], list[int]] = {}

    def append_and_trim(self, lens_type: str, prompt_id: str, variant: str, token_id: int) -> list[int]:
        key = (lens_type, prompt_id, variant)
        wl = self.windows.setdefault(key, [])
        wl.append(int(token_id))
        if len(wl) > self.window_k:
            wl.pop(0)
        return wl.copy()

    def reset_variant(self, prompt_id: str, variant: str):
        for lens in ("norm", "prism"):
            self.windows.pop((lens, prompt_id, variant), None)
```

Target API (new file: `layers_core/windows.py`)
```python
from typing import Dict, List, Tuple

class WindowManager:
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
        for lens in ("norm", "prism"):
            self.windows.pop((lens, prompt_id, variant), None)
```
Also export it in `layers_core/__init__.py` for `from layers_core import WindowManager` style.

Call‑site changes (localized)
- Remove local class definition in `run.py`.
- Add import: `from layers_core.windows import WindowManager` (or via `layers_core.__init__`).
- No other code changes required; constructors and method calls remain identical.

Validation Plan for Step 1
- Add a focused unit test `001_layers_baseline/tests/test_windows.py`:
  - `append_and_trim` maintains max length `k` and returns a copy.
  - Independent windows per `(lens, prompt_id, variant)`; no cross‑contamination.
  - `reset_variant` clears both `norm` and `prism` windows for the variant only.
- Run CPU‑only test suite: `source venv/bin/activate && scripts/run_cpu_tests.sh`.
- Ad‑hoc grep to ensure only `run.py` references the class: `rg -n "WindowManager"`.
- Optional: small smoke import to ensure import path works:
  - `python - <<'PY'\nfrom layers_core.windows import WindowManager; w=WindowManager(2); print(w.append_and_trim('norm','pos','orig',1))\nPY`

Rollback Plan
- If a test fails or unexpected behavior is observed, revert the import and restore the inline class; no data shape changes are involved.

### Step 1 Status: Completed

- Implementation:
  - Added `layers_core/windows.py` with `WindowManager` (same API and behavior).
  - Exported via `layers_core/__init__.py`.
  - Removed inline class from `run.py` and imported `WindowManager`.
  - Added unit test `001_layers_baseline/tests/test_windows.py`.
- Validation:
  - Ran `bash scripts/test.sh` — CPU-only suite passed.
  - Ran `python 001_layers_baseline/tests/test_windows.py` — passes (silent success by design).
- Behavior change: none; only code relocation and test addition.

## Next Candidate Moves (after Step 1)

Step 2: Extract head transforms detection
- Introduce `layers_core/head_transforms.py: detect_head_transforms(model)`.
- Replace local `_detect_head_transforms` in `run.py` and add a tiny unit test with a mock `cfg`/model.

Step 3: Unembedding helpers
- Introduce `layers_core/unembed.py` with weight prep and `unembed_mm`.
- Update `run.py` to call the helper; keep behavior identical (fp32 shadow promotion policy unchanged).
- Add unit tests exercising dtype promotion and device cache path using small toy tensors.

Step 4: Record builders
- Introduce `layers_core/records.py` with pure builders for per‑position and pure next‑token row dicts.
- Update call‑sites to build rows via the helpers and append; behavior guarded by CSV writer tests.

Step 5+: Pure next‑token emitter and Prism sidecar blocks
- After the supporting utilities are in place, lift `emit_pure_next_token_record(...)` and sidecar emitters into `layers_core`.
- Add targeted tests for control margin, cosine, and raw‑vs‑norm sampling flags using small fake tensors.

## Notes on Safety and Backward Compatibility

- We will not change CLI, outputs, or schema in early steps; the JSON/CSV writers and file rotation stay untouched.
- Each extraction will keep function/class names and signatures stable at call sites; only imports move.
- Where behavior must remain byte‑identical (e.g., `rest_mass` rounding), we will reuse existing helpers.
- We will avoid refactoring the multi‑pass orchestration until leaf utilities are stabilized and tested.
