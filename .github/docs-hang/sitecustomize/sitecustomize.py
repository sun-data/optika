"""Let the docs-hang diagnostic ask this Python process for its stacks.

Every Python process the docs build starts imports this module, so sending it
``SIGUSR1`` writes the stack of each of its threads to a file named after its
pid, outside the reach of the output capture of Jupyter kernels.
"""

import faulthandler
import os
import signal

__all__ = []

if "DOCS_HANG_DIR" in os.environ:
    _file = open(
        os.path.join(os.environ["DOCS_HANG_DIR"], f"{os.getpid()}.txt"),
        mode="w",
    )
    """The file this process writes its stacks to."""

    faulthandler.register(signal.SIGUSR1, file=_file, all_threads=True)
