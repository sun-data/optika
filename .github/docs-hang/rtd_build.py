"""Build the docs on Read the Docs, and report where the build hangs.

Since 2026-11-14, optika builds on Read the Docs either finish in about ten
minutes or are killed at the time limit, and a killed build keeps no output.
This script runs the same Sphinx command as Read the Docs, prefixes each line
of its output with the elapsed time, and reports the memory and CPU of the
build once a minute. If Sphinx is still running at the deadline, it dumps the
stacks of every Python process the build started and stops the build, so that
Read the Docs keeps the output.
"""

import collections
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import typing

__all__ = []

DEADLINE = int(os.environ.get("DOCS_HANG_DEADLINE", 1100))
"""Seconds to let Sphinx run before dumping its stacks and stopping it."""

HEARTBEAT = int(os.environ.get("DOCS_HANG_HEARTBEAT", 60))
"""Seconds between reports of the memory, CPU and processes of the build."""

ROUNDS = 2
"""How many times to dump the stacks at the deadline."""

PAUSE = int(os.environ.get("DOCS_HANG_PAUSE", 30))
"""Seconds between the rounds of stack dumps."""

T0 = time.monotonic()
"""When this script started."""

TICK = os.sysconf("SC_CLK_TCK")
"""Clock ticks per second, the unit of the CPU times in ``/proc``."""


def say(message: str) -> None:
    """Print a message prefixed with the seconds since this script started."""
    print(f"{time.monotonic() - T0:7.1f} {message}", flush=True)


def read(path: str | pathlib.Path) -> str:
    """Return the contents of a file, or an empty string if it can't be read."""
    try:
        return pathlib.Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def stat(path: str) -> list[str]:
    """Return the fields of a ``stat`` file after the command name."""
    text = read(path)
    if not text:
        return []
    return text.rsplit(")", 1)[1].split()


def descendants(root: int) -> list[int]:
    """Return the pids of every process descended from ``root``."""
    children = collections.defaultdict(list)
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            fields = stat(f"/proc/{entry}/stat")
            if fields:
                children[int(fields[1])].append(int(entry))
    result = []
    stack = [root]
    while stack:
        for child in children[stack.pop()]:
            result.append(child)
            stack.append(child)
    return sorted(result)


def describe(pid: int) -> str:
    """Return the state, CPU time, memory and command line of a process."""
    fields = stat(f"/proc/{pid}/stat")
    if not fields:
        return f"pid {pid} gone"
    status = {}
    for line in read(f"/proc/{pid}/status").splitlines():
        key, _, value = line.partition(":")
        status[key] = value.strip()
    cpu = (int(fields[11]) + int(fields[12])) / TICK
    command = read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()
    return (
        f"pid {pid} ppid {fields[1]} state {fields[0]} cpu {cpu:.1f} s "
        f"rss {status.get('VmRSS', '?')} threads {status.get('Threads', '?')} "
        f"wchan {read(f'/proc/{pid}/wchan') or '?'} {command[:200]}"
    )


def threads(pid: int) -> str:
    """Count the threads of a process by state and kernel wait channel."""
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return "gone"
    counts = collections.Counter()
    for tid in tids:
        fields = stat(f"/proc/{pid}/task/{tid}/stat")
        state = fields[0] if fields else "?"
        wchan = read(f"/proc/{pid}/task/{tid}/wchan") or "?"
        counts[state, wchan] += 1
    return ", ".join(f"{n} {s}/{w}" for (s, w), n in counts.most_common())


def resources() -> str:
    """Return the memory and CPU use of the build container."""
    meminfo = {}
    for line in read("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        meminfo[key] = value.strip()
    parts = [
        f"MemTotal {meminfo.get('MemTotal', '?')}",
        f"MemAvailable {meminfo.get('MemAvailable', '?')}",
        f"loadavg {read('/proc/loadavg')}",
    ]
    for name in [
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "cpu.max",
        "cpu.stat",
        "memory/memory.usage_in_bytes",
        "memory/memory.max_usage_in_bytes",
        "memory/memory.limit_in_bytes",
        "memory/memory.oom_control",
        "cpu/cpu.cfs_quota_us",
        "cpu/cpu.cfs_period_us",
        "cpu/cpu.stat",
    ]:
        value = read(f"/sys/fs/cgroup/{name}")
        if value:
            parts.append(f"{name}: {' '.join(value.split())}")
    return "; ".join(parts)


def report() -> None:
    """Print the resources of the build and every process it started."""
    say(resources())
    for pid in descendants(os.getpid()):
        say(f"  {describe(pid)}")


def dump(dumps: pathlib.Path, offsets: dict[pathlib.Path, int]) -> None:
    """Ask every Python process of the build for its stacks, and print them."""
    pids = descendants(os.getpid())
    for pid in pids:
        say(f"  {describe(pid)}")
        say(f"    threads: {threads(pid)}")
    # Only processes that imported our sitecustomize have a file, and only
    # they handle SIGUSR1; it would kill any other process.
    for path in sorted(dumps.glob("*.txt")):
        if int(path.stem) in pids:
            try:
                os.kill(int(path.stem), signal.SIGUSR1)
            except ProcessLookupError:
                pass
    time.sleep(5)
    for path in sorted(dumps.glob("*.txt")):
        text = path.read_text(errors="replace")
        new = text[offsets.get(path, 0) :]
        offsets[path] = len(text)
        if new.strip():
            command = read(f"/proc/{path.stem}/cmdline").replace("\0", " ")
            say(f"STACKS of pid {path.stem}: {command[:200]}")
            for line in new.splitlines():
                say(f"    {line}")


def echo(stream: typing.IO[str]) -> None:
    """Print each line of a stream, prefixed with the elapsed time."""
    for line in stream:
        say(line.rstrip("\n"))


def main() -> None:
    """Run Sphinx on the docs, and dump its stacks if it outlives the deadline.

    Any arguments replace the Sphinx command, which tests this script.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    here = pathlib.Path(__file__).resolve().parent
    dumps = pathlib.Path(tempfile.mkdtemp(prefix="docs-hang-"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(here / "sitecustomize"), env.get("PYTHONPATH")])
    )
    env["DOCS_HANG_DIR"] = str(dumps)
    env["PYTHONUNBUFFERED"] = "1"

    say(f"cpu_count {os.cpu_count()}, affinity {len(os.sched_getaffinity(0))}")
    say(f"uname {' '.join(os.uname())}")
    report()

    command = sys.argv[1:] or [
        sys.executable,
        "-m",
        "sphinx",
        "-T",
        "-b",
        "html",
        "-d",
        "_build/doctrees",
        "-D",
        "language=en",
        ".",
        os.path.join(os.environ["READTHEDOCS_OUTPUT"], "html"),
    ]
    say(f"running {' '.join(command)}")
    sphinx = subprocess.Popen(
        command,
        cwd=root / "docs",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    reader = threading.Thread(target=echo, args=(sphinx.stdout,), daemon=True)
    reader.start()

    hung = False
    while sphinx.poll() is None:
        try:
            sphinx.wait(timeout=HEARTBEAT)
        except subprocess.TimeoutExpired:
            report()
            if time.monotonic() - T0 > DEADLINE:
                hung = True
                break

    if hung:
        say(f"DEADLINE: Sphinx is still running after {DEADLINE} s")
        offsets = {}
        for i in range(ROUNDS):
            say(f"STACK DUMP {i + 1} of {ROUNDS}")
            dump(dumps, offsets)
            if i + 1 < ROUNDS:
                time.sleep(PAUSE)
        pids = descendants(os.getpid())
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        sphinx.wait()

    reader.join(timeout=10)
    report()
    say(f"EXIT {sphinx.returncode}")
    sys.exit(1 if hung else sphinx.returncode)


if __name__ == "__main__":
    main()
