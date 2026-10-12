from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from wsl.data.cache import IntegrityError, RawCache, sha256_bytes


def test_store_and_read_roundtrip(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    res = cache.store("src/a", b"hello", url="https://x/a", ext="txt")
    assert res.new_content and res.pinned and res.previous_pin is None
    content, rec = cache.read("src/a")
    assert content == b"hello"
    assert rec.sha256 == sha256_bytes(b"hello")
    assert (tmp_path / "data/manifest.json").exists()


def test_files_are_read_only(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    rec = cache.store("src/a", b"x", url="u", ext="txt").record
    mode = os.stat(tmp_path / rec.path).st_mode
    assert not mode & stat.S_IWUSR


def test_same_bytes_are_not_duplicated(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    cache.store("src/a", b"x", url="u", ext="txt")
    res = cache.store("src/a", b"x", url="u", ext="txt")
    assert not res.new_content
    assert len(cache.entry("src/a")["retrievals"]) == 1  # type: ignore[index]


def test_changed_source_is_not_silently_repinned(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    cache.store("src/a", b"v1", url="u", ext="txt")
    res = cache.store("src/a", b"v2", url="u", ext="txt")
    assert res.new_content and not res.pinned
    assert cache.read("src/a")[0] == b"v1"
    res = cache.store("src/a", b"v2", url="u", ext="txt", repin=True)
    assert res.pinned
    assert cache.read("src/a")[0] == b"v2"


def test_tampering_is_detected(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    rec = cache.store("src/a", b"original", url="u", ext="txt").record
    path = tmp_path / rec.path
    os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
    path.write_bytes(b"modified")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        cache.read("src/a")


def test_manifest_survives_reload(tmp_path: Path) -> None:
    RawCache(root=tmp_path).store("src/a", b"x", url="u", ext="txt")
    assert RawCache(root=tmp_path).read("src/a")[0] == b"x"


def test_missing_local_file_fails_loudly(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    rec = cache.store("src/a", b"x", url="u", ext="txt").record
    path = tmp_path / rec.path
    os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
    path.unlink()
    with pytest.raises(IntegrityError, match="not in the local cache"):
        RawCache(root=tmp_path).read("src/a")


@pytest.mark.parametrize("bad", ["", "/abs", "a/../b"])
def test_rejects_unsafe_keys(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        RawCache(root=tmp_path).store(bad, b"x", url="u", ext="txt")


def test_drop_removes_entries_but_keeps_files(tmp_path: Path) -> None:
    cache = RawCache(root=tmp_path)
    rec = cache.store("src/a", b"x", url="u", ext="txt").record
    assert cache.drop({"src/a", "src/never"}) == ["src/a"]
    assert cache.entry("src/a") is None
    assert (tmp_path / rec.path).exists()
