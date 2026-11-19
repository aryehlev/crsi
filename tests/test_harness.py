import os

import pytest

from crsi import load_harness
from crsi.harness import apply_edits, changed_files, copy_harness, diff_files, fingerprint, read_files, safe_path, syntax_errors


def test_read_copy_and_diff(tmp_path):
    src = tmp_path / "h"
    (src / "pkg").mkdir(parents=True)
    (src / "harness.py").write_text("def run(task, llm):\n    return 'a'\n")
    (src / "pkg" / "notes.txt").write_text("hello\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
    (src / "blob.bin").write_bytes(b"\xff\xfe\x00")
    files = read_files(src)
    assert list(files) == ["harness.py", "pkg/notes.txt"]  # caches and binary files are left out

    copy_harness(src, tmp_path / "copy")
    assert not (tmp_path / "copy" / "__pycache__").exists() and (tmp_path / "copy" / "blob.bin").exists()
    apply_edits(tmp_path / "copy", {"harness.py": "def run(task, llm):\n    return 'b'\n", "new.py": "X = 1\n"}, ["pkg/notes.txt"])
    new = read_files(tmp_path / "copy")
    assert changed_files(files, new) == ["harness.py", "new.py", "pkg/notes.txt"]
    diff = diff_files(files, new)
    assert "-    return 'a'" in diff and "+    return 'b'" in diff and "+++ b/new.py" in diff and "+++ /dev/null" in diff
    assert diff_files(files, files) == "" and fingerprint(files) != fingerprint(new)


@pytest.mark.parametrize("path", ["../escape.py", "/etc/passwd", "a/../../b.py", "", "__pycache__/x.py", "C:/x.py"])
def test_paths_named_by_a_model_cannot_leave_the_harness(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)
    with pytest.raises(ValueError):
        apply_edits(tmp_path, {path: "x"})
    assert not list(tmp_path.iterdir())


def test_two_harnesses_with_the_same_module_names_do_not_share_code(tmp_path):
    for name, value in (("one", 1), ("two", 2)):
        root = tmp_path / name
        root.mkdir()
        (root / "helpers.py").write_text(f"VALUE = {value}\n")
        (root / "harness.py").write_text("from . import helpers\n\ndef run(task, llm):\n    return helpers.VALUE\n")
    one, two = load_harness(tmp_path / "one"), load_harness(tmp_path / "two")
    assert (one.run(None, None), two.run(None, None)) == (1, 2)
    assert load_harness(tmp_path / "one") is one  # cached while unchanged


def test_a_changed_harness_is_reloaded(tmp_path):
    (tmp_path / "harness.py").write_text("def run(task, llm):\n    return 'old'\n")
    assert load_harness(tmp_path).run(None, None) == "old"
    stat = (tmp_path / "harness.py").stat()
    # Same size, same timestamp: the case where Python's own bytecode cache would serve stale code.
    (tmp_path / "harness.py").write_text("def run(task, llm):\n    return 'new'\n")
    os.utime(tmp_path / "harness.py", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert load_harness(tmp_path).run(None, None) == "new"
    assert not (tmp_path / "__pycache__").exists()
    with pytest.raises(FileNotFoundError):
        load_harness(tmp_path / "missing")


def test_syntax_errors_are_reported_per_file():
    errors = syntax_errors({"ok.py": "x = 1\n", "bad.py": "def run(:\n", "notes.txt": "def run(:"})
    assert len(errors) == 1 and errors[0].startswith("bad.py:1:")
