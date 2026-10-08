import os
import hashlib
import pathlib
import importlib.metadata
import joblib

__all__ = [
    "memory",
]


def _version() -> str:
    """The installed version of optika, or ``"unknown"`` if it is not installed."""
    try:
        return importlib.metadata.version("optika")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _fingerprint(
    root: pathlib.Path,
) -> str:
    """
    A fingerprint of the files of a package: a hash of the path, size and
    modification time of each, except its bytecode caches and its tests,
    which do not change what it computes.

    Checking the modification times rather than reading the files keeps this
    fast enough to run on import, since the package holds many megabytes of
    optical constants.

    Parameters
    ----------
    root
        The directory of the package.
    """
    result = hashlib.sha256()
    for directory, names_directory, names_file in os.walk(root):
        names_directory[:] = sorted(
            name for name in names_directory if name not in ("__pycache__", "_tests")
        )
        for name in sorted(names_file):
            if name.endswith("_test.py"):
                continue
            path = pathlib.Path(directory) / name
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            result.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return result.hexdigest()[:16]


_path_cache = (
    pathlib.Path.home()
    / ".optika/cache"
    / f"{_version()}-{_fingerprint(pathlib.Path(__file__).parent)}"
)

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""
A representation of the cache which stores intermediate results.

Each entry is keyed on the source of the function that computed it,
but not on the code that function calls or the data it reads,
so the cache lives in a directory of ``~/.optika/cache`` named after the
version of optika and a fingerprint of its files.
A change to any of them starts a new cache,
so a result is never returned by a different version of optika's code or
data than the one that computed it.
Old directories are not removed, and can be deleted at any time.
"""
