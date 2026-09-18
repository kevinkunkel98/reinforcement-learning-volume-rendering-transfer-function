"""What produced a result file: the commit, the working tree's cleanliness,
and a fingerprint of the code that actually computed the score.

Written after a batch evaluation started in the evening, a scoring fix landed
in `goals.py`/`visibility.py` at 22:48 while it ran, and the job wrote its
files at 03:26 -- carrying numbers from the code it had imported hours
earlier. The files' timestamps were newer than the fix, so nothing about them
looked wrong, and a week of thesis numbers (a policy attainment of +0.169
where the corrected figure is +0.275, plus a null result drawn from the same
batch) was published from them.

Two pieces make that visible. `provenance()` records the code identity, and
`IMPORT_TIME_PROVENANCE` freezes it at import -- a long-running job must
report the code it loaded, not what git happens to say when it finally
writes. `compare()` then checks a recorded block against the code running
now, so a stale result announces itself instead of being trusted.

Standard library only, and every git call degrades to `None`: a research
script must never die because provenance could not be read.
"""
import datetime
import hashlib
import importlib.util
import os
import subprocess
import sys

# The modules that actually turn a rendered state into a number. A change in
# any of them invalidates comparisons across result files, which is exactly
# what the incident's numbers hid.
DEFAULT_SCORING_MODULES = ("goals", "visibility", "rl.vis_eval", "rl.oneshot_env")

FINGERPRINT_CHARS = 12
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_GIT_TIMEOUT_SECONDS = 10


# --- git ---------------------------------------------------------------------

def _git(*args) -> str | None:
    """`git *args` in this repo's directory, or `None` if that cannot be
    answered -- git missing, not a checkout, or the call failing for any
    other reason. Never raises: provenance is a note on the side of a
    computation, never a reason to lose one."""
    try:
        completed = subprocess.run(["git", "-C", _REPO_ROOT, *args],
                                    capture_output=True, text=True,
                                    timeout=_GIT_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_commit() -> str | None:
    """The full SHA of HEAD, or `None` when git cannot answer."""
    out = _git("rev-parse", "HEAD")
    return out.strip() if out else None


def git_dirty() -> bool | None:
    """Whether tracked files differ from HEAD, or `None` when git cannot
    answer. Untracked files are deliberately excluded: evaluation jobs drop
    new files into `out/` constantly, and a run flagged dirty by its own
    output would train everyone to ignore the flag."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    if out is None:
        return None
    return bool(out.strip())


# --- fingerprinting the scoring code -------------------------------------------

def _module_file(module) -> tuple[str, str | None]:
    """`(name, path)` for a module object or a dotted module name. Already
    imported modules are read through their own `__file__`, so the
    fingerprint describes the code this process loaded; a module that is not
    imported is located through its spec rather than imported for the sake of
    a hash (importing `rl.vis_eval` from here would be circular anyway)."""
    if not isinstance(module, str):
        return getattr(module, "__name__", repr(module)), getattr(module, "__file__", None)

    loaded = sys.modules.get(module)
    if loaded is not None:
        return module, getattr(loaded, "__file__", None)
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        return module, None
    return module, getattr(spec, "origin", None) if spec else None


def _read_source(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return None


def scoring_fingerprint(modules=None) -> str:
    """First `FINGERPRINT_CHARS` hex characters of a sha256 over the source
    text of `modules` (default: `DEFAULT_SCORING_MODULES`), read from disk.

    The module's name and its source length go into the hash alongside the
    source, so neither reordering the list nor a source that happens to end
    where another begins can collide. A module whose source cannot be read
    hashes as unreadable rather than as empty -- "I could not tell" and "the
    file is empty" must not produce the same fingerprint."""
    digest = hashlib.sha256()
    for module in (DEFAULT_SCORING_MODULES if modules is None else modules):
        name, path = _module_file(module)
        source = _read_source(path)
        body = "<unreadable>" if source is None else source
        digest.update(f"{name}\n{len(body)}\n".encode("utf-8"))
        digest.update(body.encode("utf-8"))
    return digest.hexdigest()[:FINGERPRINT_CHARS]


# --- the record ------------------------------------------------------------------

def provenance(modules=None) -> dict:
    """A record of what is about to compute (or has just computed) a result:
    the commit, whether the tree was dirty, the scoring fingerprint, the
    interpreter and the capture time."""
    names = [name for name, _ in
             (_module_file(module) for module in
              (DEFAULT_SCORING_MODULES if modules is None else modules))]
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "scoring_fingerprint": scoring_fingerprint(modules),
        "scoring_modules": names,
        "python": sys.version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# Frozen at import, which is the only moment at which "the code on disk" and
# "the code this process is running" are known to agree. Writers record this,
# not a fresh call at write time -- the whole incident is the gap between the
# two.
IMPORT_TIME_PROVENANCE = provenance()


# --- staleness -------------------------------------------------------------------

def _short(value, length: int = 12) -> str:
    return "unknown" if value is None else str(value)[:length]


def compare(recorded: dict, current: dict = None) -> dict:
    """Check a result file's `provenance` block against the code running now.

    Returns `{"stale", "reasons", "recorded_dirty"}`. Stale means the numbers
    were produced by code that is no longer what this checkout runs: a
    different scoring fingerprint or a different commit. Fields that either
    side could not record (git absent, an old result file) are not guessed at
    -- an unknown is reported as such rather than as a difference. A result
    with no provenance at all is stale by definition: that is precisely the
    file nobody could vet.

    `recorded_dirty` is not staleness -- the code may match exactly -- but a
    run made from an uncommitted tree cannot be reproduced from its commit,
    so the caller should say so out loud."""
    current = provenance() if current is None else current

    if not recorded:
        return {"stale": True,
                "reasons": ["no provenance recorded: this result cannot be traced to "
                            "the code that produced it"],
                "recorded_dirty": False}

    reasons = []
    recorded_fingerprint = recorded.get("scoring_fingerprint")
    current_fingerprint = current.get("scoring_fingerprint")
    if (recorded_fingerprint and current_fingerprint
            and recorded_fingerprint != current_fingerprint):
        reasons.append("scoring code changed: fingerprint "
                       f"{recorded_fingerprint} -> {current_fingerprint}")
    elif not recorded_fingerprint:
        reasons.append("no scoring fingerprint recorded: the numbers cannot be "
                       "attributed to any version of the scoring code")

    recorded_commit = recorded.get("git_commit")
    current_commit = current.get("git_commit")
    if recorded_commit and current_commit and recorded_commit != current_commit:
        reasons.append(f"git commit changed: {_short(recorded_commit, 8)} -> "
                       f"{_short(current_commit, 8)}")

    return {"stale": bool(reasons), "reasons": reasons,
            "recorded_dirty": bool(recorded.get("git_dirty"))}


_BANNER_WIDTH = 78


def format_staleness(report: dict) -> str:
    """A banner for a stale `compare()` report, or `""` when the result still
    matches the running code. Loud on purpose: the failure this guards
    against was invisible precisely because everything about the file looked
    ordinary.

    A dirty tree alone does not raise the banner -- during development the
    tree is dirty nearly always, and a warning that fires on every run is a
    warning nobody reads. It is reported as an extra line once something else
    is already wrong."""
    if not report.get("stale"):
        return ""

    lines = list(report.get("reasons", []))
    if report.get("recorded_dirty"):
        lines.append("and the recorded run had uncommitted changes to tracked files, "
                     "so its commit does not describe the code it ran")

    rule = "!" * _BANNER_WIDTH
    head = "!!! STALE RESULT"
    body = "\n".join(f"!!!   {line}" for line in lines)
    tail = ("!!!   These numbers were not produced by the code in this checkout. "
            "Re-run\n!!!   before quoting them.")
    return f"{rule}\n{head}\n{body}\n{tail}\n{rule}"
