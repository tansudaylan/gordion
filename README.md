# Gordion

The former TESS pixel-variability workflow now lives in [Lygos](../lygos). This repository is reserved for future work.

## Runtime paths

Install the package and set its repository path before using repository-local data or visualization directories:

```bash
python -m pip install -e .
export GORDION_PATH=/path/to/gordion
```

`GORDION_PATH` identifies the repository root. Runtime inputs and generated figures belong in `data/` and `visuals/`, which are ignored by Git.
