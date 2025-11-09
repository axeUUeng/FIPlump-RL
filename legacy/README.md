# Legacy code museum

This folder preserves the original experimentation code that pre-dates the `plump_rl` environment:

- `Utilis/` and `Utilis.py`: early card/hand/player abstractions plus the Hearts environment.
- `PlumpEnv.py` / `PLUMP_ENV.py`: first drafts of a Plump Gym wrapper.
- `test.ipynb`, `advice.txt`, and related artifacts from the initial spike.

Nothing under `legacy/` is imported by the current RL environment; keep it around only for historical reference or manual inspection. New development should target the modules in `plump_rl/`.
