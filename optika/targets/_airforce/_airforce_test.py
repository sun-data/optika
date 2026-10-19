import subprocess
import sys
import optika


def test_airforce():
    result = optika.targets.airforce("x", "y")
    assert result.ndim == 2
    assert result.sum() > 0


def test_import_without_pymupdf():
    """
    Importing optika does not import PyMuPDF, which imports ONNX Runtime, so
    only those who load the target pay for it.
    """
    code = "import sys, optika; sys.exit('pymupdf' in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
