import os
import hashlib
import pathlib
import importlib.metadata
from typing import Iterator
import joblib

__all__ = [
    "memory",
]

_dependencies = (
    "named-arrays",
    "numpy",
    "scipy",
    "astropy",
)
"""The packages, besides optika, whose code computes the cached results."""

_suffixes_data = (
    ".nk",
    ".txt",
    ".csv",
    ".dat",
    ".svg",
)
"""The kinds of data files in optika, matched without regard to case."""


def _version(
    name: str,
) -> str:
    """
    The installed version of a package, or ``"unknown"`` if it is not installed.

    Parameters
    ----------
    name
        The name of the package.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _files(
    directory: str,
    prefix: str = "",
) -> Iterator[str]:
    """
    A line for each file below a directory that can change what optika
    computes: the hash of the contents of each module,
    and the size and modification time of each data file.

    Hidden files, bytecode caches, tests, files of any other kind,
    and files that cannot be read, such as broken links, are skipped.

    Parameters
    ----------
    directory
        The directory to search.
    prefix
        The path of `directory` relative to the package,
        with a trailing slash.
    """
    with os.scandir(directory) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        name = entry.name
        path = f"{prefix}{name}"
        if name.startswith(".") or name in ("__pycache__", "_tests"):
            continue
        try:
            if entry.is_dir():
                yield from _files(entry.path, f"{path}/")
            elif name.endswith("_test.py"):
                continue
            elif name.endswith(".py"):
                with open(entry.path, "rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                yield f"{path}\0{digest}\n"
            elif name.lower().endswith(_suffixes_data):
                stat = entry.stat()
                yield f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        except OSError:
            continue


def _fingerprint(
    root: pathlib.Path,
) -> str:
    """
    A fingerprint of the code and data which compute the cached results:
    a hash of the versions of the dependencies of a package,
    the contents of each of its modules,
    and the size and modification time of each of its data files.

    Hashing the contents of the modules lets identical code share a cache,
    after a reinstall or a switch to another branch and back.
    The data files hold many megabytes of optical constants,
    too many to read on every import.

    Parameters
    ----------
    root
        The directory of the package.
    """
    result = hashlib.sha256()
    for name in _dependencies:
        result.update(f"{name}\0{_version(name)}\n".encode())
    for line in _files(str(root)):
        result.update(os.fsencode(line))
    return result.hexdigest()[:16]


_path_cache = (
    pathlib.Path.home() / ".optika/cache" / _fingerprint(pathlib.Path(__file__).parent)
)

memory = joblib.Memory(location=_path_cache, mmap_mode="r", verbose=0)
"""
A representation of the cache which stores intermediate results.

Each entry is keyed on the source of the function that computed it,
but not on the code that function calls or the data it reads,
so the cache lives in a directory of ``~/.optika/cache`` named after a
fingerprint of the code and data of optika and the versions of its
dependencies.
A change to any of them starts a new cache.
An edit to an editable install of a dependency, such as named-arrays,
does not change its version, so clear the cache with
``optika.memory.clear()`` after one.
Old directories are not removed, and can be deleted at any time.
"""
