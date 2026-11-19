"""A harness is a directory of files. This module reads, copies, diffs and imports one."""

from __future__ import annotations

import difflib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import shutil
import sys
import threading
from pathlib import Path, PurePosixPath
from types import ModuleType

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILES = {".DS_Store"}
MAX_TEXT_BYTES = 400_000


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_FILES or path.suffix == ".pyc":
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def read_files(harness_dir: str | Path) -> dict[str, str]:
    """The harness's text files, keyed by posix path. Binary and very large files are left out."""
    root = Path(harness_dir)
    out: dict[str, str] = {}
    for path in _files(root):
        if path.stat().st_size > MAX_TEXT_BYTES:
            continue
        try:
            out[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return out


def copy_harness(src: str | Path, dst: str | Path) -> None:
    """Replace `dst` with a copy of `src`."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"harness directory not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP_DIRS, *SKIP_FILES, "*.pyc"))


def safe_path(root: str | Path, relative: str) -> Path:
    """Resolve a path named by a model, refusing anything that leaves the harness directory."""
    rel = PurePosixPath(str(relative).strip().replace("\\", "/"))
    if not rel.parts or rel.is_absolute() or ".." in rel.parts or rel.parts[0].endswith(":"):
        raise ValueError(f"unsafe path: {relative!r}")
    if any(part in SKIP_DIRS for part in rel.parts):
        raise ValueError(f"path not allowed in a harness: {relative!r}")
    return Path(root).joinpath(*rel.parts)


def apply_edits(harness_dir: str | Path, files: dict[str, str], deletes: list[str] | None = None) -> None:
    """Write whole-file replacements and deletions into a harness directory."""
    root = Path(harness_dir)
    targets = {name: safe_path(root, name) for name in files}
    removals = [safe_path(root, name) for name in deletes or []]
    for name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(files[name], encoding="utf-8")
    for target in removals:
        if target.is_file():
            target.unlink()


def diff_files(old: dict[str, str], new: dict[str, str]) -> str:
    """A unified diff between two harnesses. Empty when they are the same."""
    chunks: list[str] = []
    for name in sorted(set(old) | set(new)):
        before, after = old.get(name), new.get(name)
        if before == after:
            continue
        lines = difflib.unified_diff(
            (before or "").splitlines(),
            (after or "").splitlines(),
            fromfile=f"a/{name}" if before is not None else "/dev/null",
            tofile=f"b/{name}" if after is not None else "/dev/null",
            lineterm="",
        )
        body = "\n".join(lines)
        # A change that is only a trailing newline produces no hunks; still report the file.
        chunks.append(body or f"--- a/{name}\n+++ b/{name}\n(whitespace-only change)")
    return "\n".join(chunks)


def changed_files(old: dict[str, str], new: dict[str, str]) -> list[str]:
    return [name for name in sorted(set(old) | set(new)) if old.get(name) != new.get(name)]


def fingerprint(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(files[name].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def syntax_errors(files: dict[str, str]) -> list[str]:
    errors = []
    for name in sorted(files):
        if not name.endswith(".py"):
            continue
        try:
            compile(files[name], name, "exec")
        except SyntaxError as exc:
            errors.append(f"{name}:{exc.lineno}: {exc.msg}")
        except ValueError as exc:  # null bytes
            errors.append(f"{name}: {exc}")
    return errors


_LOCK = threading.RLock()
_CACHE: dict[str, tuple[str, ModuleType]] = {}
_COUNTER = itertools.count()
_PREFIX = "_crsi_harness_"


class _SourceLoader(importlib.machinery.SourceFileLoader):
    """Always compile from source. Python trusts cached bytecode when a file's size and
    whole-second mtime match, which a harness rewritten in place can do."""

    def get_code(self, fullname):
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _HarnessFinder(importlib.abc.MetaPathFinder):
    """Routes the sibling modules of a loaded harness through _SourceLoader as well."""

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(_PREFIX) or path is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _SourceLoader(spec.loader.name, spec.loader.path)
        return spec


sys.meta_path.insert(0, _HarnessFinder())


def _signature(root: Path) -> str:
    return fingerprint({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace") for p in _files(root) if p.suffix == ".py"})


def _unload(name: str) -> None:
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]


def load_harness(harness_dir: str | Path, entry: str = "harness.py") -> ModuleType:
    """Import a harness directory in this process and return its entry module.

    The directory is loaded as a package under a private name, so two harnesses never share
    modules. A harness with several files imports its siblings relatively
    (`from . import tools`). The module is cached until a .py file in the directory changes.

    This executes the harness. A harness is code written by a model: run it in a sandbox.
    """
    root = Path(harness_dir).resolve()
    path = root / entry
    if not path.is_file():
        raise FileNotFoundError(f"{entry} not found in harness directory {root}")
    key = f"{root}::{entry}"
    with _LOCK:
        signature = _signature(root)
        hit = _CACHE.get(key)
        if hit and hit[0] == signature:
            return hit[1]
        if hit:
            _unload(hit[1].__name__)
            del _CACHE[key]
        name = f"{_PREFIX}{hashlib.sha1(key.encode()).hexdigest()[:10]}_{next(_COUNTER)}"
        spec = importlib.util.spec_from_file_location(
            name, path, loader=_SourceLoader(name, str(path)), submodule_search_locations=[str(root)]
        )
        if spec is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            _unload(name)
            raise
        _CACHE[key] = (signature, module)
        return module
