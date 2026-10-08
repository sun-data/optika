import os
import pathlib
import importlib.metadata
import pytest
import optika
from . import _caching


def test_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """The version is the installed one, or a placeholder if there is none."""
    assert _caching._version() == importlib.metadata.version("optika")

    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert _caching._version() == "unknown"


def test_fingerprint(
    tmp_path: pathlib.Path,
):
    """
    The fingerprint changes with any file of the package,
    but not with its bytecode caches or its tests.
    """
    code = tmp_path / "module.py"
    code.write_text("x = 1")
    data = tmp_path / "data" / "table.csv"
    data.parent.mkdir()
    data.write_text("1, 2")

    result = _caching._fingerprint(tmp_path)
    assert len(result) == 16
    assert _caching._fingerprint(tmp_path) == result

    ignored = [
        tmp_path / "__pycache__" / "module.cpython-313.pyc",
        tmp_path / "_tests" / "test_module.py",
        tmp_path / "module_test.py",
    ]
    for path in ignored:
        path.parent.mkdir(exist_ok=True)
        path.write_text("ignored")
    assert _caching._fingerprint(tmp_path) == result

    # a change in size
    data.write_text("1, 23")
    changed_size = _caching._fingerprint(tmp_path)
    assert changed_size != result

    # a change in the modification time alone
    stat = code.stat()
    os.utime(code, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    changed_time = _caching._fingerprint(tmp_path)
    assert changed_time != changed_size

    # a new file
    (tmp_path / "data" / "other.csv").write_text("3")
    assert _caching._fingerprint(tmp_path) != changed_time


def test_memory():
    """
    The cache lives in a directory named after the version of optika and a
    fingerprint of its files.
    """
    location = pathlib.Path(optika.memory.location)
    assert location.parent == pathlib.Path.home() / ".optika/cache"
    version, fingerprint = location.name.rsplit("-", 1)
    assert version == _caching._version()
    assert len(fingerprint) == 16
