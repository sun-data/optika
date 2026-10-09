import os
import pathlib
import numpy as np
import pytest
import optika
from . import _caching


def test_version() -> None:
    """The version of a package is the installed one, or a placeholder if there is none."""
    assert _caching._version("numpy") == np.__version__
    assert _caching._version("not-a-package") == "unknown"


def test_fingerprint(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The fingerprint changes with the contents of the code, the size or
    modification time of the data, a new file, or the version of a dependency,
    but not with the modification time of the code or with any other file.
    """
    code = tmp_path / "module.py"
    code.write_text("x = 1")
    data = tmp_path / "data" / "table.TXT"
    data.parent.mkdir()
    data.write_text("1, 2")

    result = _caching._fingerprint(tmp_path)
    assert len(result) == 16
    assert _caching._fingerprint(tmp_path) == result

    ignored = [
        tmp_path / "__pycache__" / "module.cpython-313.pyc",
        tmp_path / "_tests" / "test_module.py",
        tmp_path / ".ipynb_checkpoints" / "module-checkpoint.py",
        tmp_path / "module_test.py",
        tmp_path / ".#module.py",
        tmp_path / "module.py~",
        tmp_path / "tmp.png",
    ]
    for path in ignored:
        path.parent.mkdir(exist_ok=True)
        path.write_text("ignored")
    assert _caching._fingerprint(tmp_path) == result

    # the contents of the code, with the same size and modification time
    stat = code.stat()
    code.write_text("x = 2")
    os.utime(code, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed_code = _caching._fingerprint(tmp_path)
    assert changed_code != result

    # but not the modification time of the code alone
    os.utime(code, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert _caching._fingerprint(tmp_path) == changed_code

    # the size of the data
    data.write_text("1, 23")
    changed_size = _caching._fingerprint(tmp_path)
    assert changed_size != changed_code

    # the modification time of the data alone
    stat = data.stat()
    os.utime(data, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    changed_time = _caching._fingerprint(tmp_path)
    assert changed_time != changed_size

    # a new file
    (tmp_path / "data" / "other.csv").write_text("3")
    changed_file = _caching._fingerprint(tmp_path)
    assert changed_file != changed_time

    # the version of a dependency
    def version(name: str) -> str:
        return "0.0.0"

    monkeypatch.setattr(_caching, "_version", version)
    assert _caching._fingerprint(tmp_path) != changed_file


def test_fingerprint_unreadable(
    tmp_path: pathlib.Path,
) -> None:
    """Files which cannot be read, such as broken links, are skipped."""
    (tmp_path / "module.py").write_text("x = 1")
    (tmp_path / "table.csv").write_text("1, 2")
    result = _caching._fingerprint(tmp_path)

    try:
        (tmp_path / "missing.py").symlink_to(tmp_path / "nothing.py")
        (tmp_path / "missing.csv").symlink_to(tmp_path / "nothing.csv")
    except OSError:  # pragma: no cover
        pytest.skip("links cannot be created without privileges on Windows")

    assert _caching._fingerprint(tmp_path) == result


def test_memory() -> None:
    """The cache lives in a directory named after the fingerprint of optika."""
    fingerprint = _caching._fingerprint(pathlib.Path(optika.__file__).parent)
    location = pathlib.Path(optika.memory.location)
    assert location == pathlib.Path.home() / ".optika/cache" / fingerprint
