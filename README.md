# Gordion

The TESS pixel-variability workflow has moved to [Lygos](../lygos). Gordion is reserved for future work; there is no active workflow here yet.

## Runtime paths

Install the package and set its repository path before using repository-local data or visualization directories:

```bash
python -m pip install -e .
export GORDION_PATH=/path/to/gordion
```

`GORDION_PATH` identifies the repository root. Runtime inputs and generated figures belong in `data/` and `visuals/`, which are ignored by Git.
