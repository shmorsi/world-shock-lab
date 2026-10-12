"""Immutable raw-download cache with a committed SHA-256 manifest.

Every byte that enters the pipeline is stored once, read-only, under
``data/raw/<source>/<artifact>/<UTC timestamp>_<sha12>.<ext>`` and recorded in
``data/manifest.json``. Each artifact has a *pinned* hash; the build reads only pinned
content and verifies the hash on every read.

Re-fetching never silently changes the data a build uses: if a source now serves different
bytes than the pin, the new bytes are stored and reported but not pinned until the caller
asks for it (``repin=True``). A fresh clone whose download no longer matches the committed
pin therefore fails loudly instead of quietly producing different numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wsl.paths import ROOT

RAW_SUBDIR = Path("data/raw")
MANIFEST_SUBPATH = Path("data/manifest.json")


class IntegrityError(RuntimeError):
    """Raised when cached bytes do not match their recorded hash, or no pinned copy exists."""


@dataclass(frozen=True)
class RawRecord:
    key: str
    url: str
    retrieved_at: str
    sha256: str
    size: int
    path: str  # relative to the repository root
    content_type: str | None


@dataclass(frozen=True)
class StoreResult:
    record: RawRecord
    new_content: bool  # bytes never seen before for this artifact
    pinned: bool  # the stored bytes are now the pinned version
    previous_pin: str | None


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_key(key: str) -> str:
    if not key or key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"invalid artifact key: {key!r}")
    return key


class RawCache:
    def __init__(self, root: Path = ROOT, manifest: Path = MANIFEST_SUBPATH) -> None:
        self.root = root
        self.manifest_path = root / manifest
        self._manifest: dict[str, Any] = self._load()

    # -- manifest -------------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            data: dict[str, Any] = json.loads(self.manifest_path.read_text())
            return data
        return {"version": 1, "artifacts": {}}

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.manifest_path)

    @property
    def keys(self) -> list[str]:
        return sorted(self._manifest["artifacts"])

    def entry(self, key: str) -> dict[str, Any] | None:
        entry: dict[str, Any] | None = self._manifest["artifacts"].get(key)
        return entry

    # -- write ----------------------------------------------------------------------------
    def store(
        self,
        key: str,
        content: bytes,
        *,
        url: str,
        ext: str,
        content_type: str | None = None,
        repin: bool = False,
    ) -> StoreResult:
        key = _safe_key(key)
        digest = sha256_bytes(content)
        entry = self._manifest["artifacts"].setdefault(key, {"pinned": None, "retrievals": []})
        previous_pin: str | None = entry["pinned"]

        existing = next((r for r in entry["retrievals"] if r["sha256"] == digest), None)
        if existing is not None and (self.root / existing["path"]).exists():
            record = RawRecord(key=key, **existing)
            new_content = False
        else:
            stamp = _utc_now()
            rel = RAW_SUBDIR / key / f"{stamp.replace(':', '')}_{digest[:12]}.{ext}"
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)
                os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only
            record = RawRecord(
                key=key,
                url=url,
                retrieved_at=stamp,
                sha256=digest,
                size=len(content),
                path=str(rel),
                content_type=content_type,
            )
            if existing is None:
                entry["retrievals"].append({k: v for k, v in asdict(record).items() if k != "key"})
            new_content = existing is None

        pinned = False
        if previous_pin is None or repin:
            entry["pinned"] = digest
            pinned = True
        elif previous_pin == digest:
            pinned = True
        self._save()
        return StoreResult(record=record, new_content=new_content, pinned=pinned,
                           previous_pin=previous_pin)

    def drop(self, keys: set[str]) -> list[str]:
        """Remove manifest entries (raw files stay on disk). Returns the keys removed."""
        removed = sorted(k for k in keys if k in self._manifest["artifacts"])
        for k in removed:
            del self._manifest["artifacts"][k]
        if removed:
            self._save()
        return removed

    # -- read -----------------------------------------------------------------------------
    def pinned_record(self, key: str) -> RawRecord:
        entry = self.entry(key)
        if entry is None or entry["pinned"] is None:
            raise IntegrityError(f"no pinned artifact for {key!r}; run `wsl data fetch` first")
        rec = next((r for r in entry["retrievals"] if r["sha256"] == entry["pinned"]), None)
        if rec is None:
            raise IntegrityError(f"manifest for {key!r} pins a hash with no retrieval record")
        return RawRecord(key=key, **rec)

    def read(self, key: str) -> tuple[bytes, RawRecord]:
        record = self.pinned_record(key)
        path = self.root / record.path
        if not path.exists():
            raise IntegrityError(
                f"pinned file for {key!r} is not in the local cache ({record.path}). "
                "Run `wsl data fetch`; if the source has since revised the file, the fetch "
                "will say so and `--repin` adopts the new version."
            )
        content = path.read_bytes()
        if sha256_bytes(content) != record.sha256:
            raise IntegrityError(f"hash mismatch for {key!r}: cached file was modified")
        return content, record

    def has_pinned_locally(self, key: str) -> bool:
        try:
            self.read(key)
        except IntegrityError:
            return False
        return True
