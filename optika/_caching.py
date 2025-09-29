import pathlib
import joblib

__all__ = [
    "memory",
]

_path_cache = pathlib.Path.home() / ".optika/cache"

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""A representation of the cache which stores intermediate results."""
