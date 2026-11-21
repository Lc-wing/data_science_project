"""HTTP download helper with retries, and the provenance log for raw snapshots."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from macao_tourism.config import PROJECT_ROOT, USER_AGENT, sha256_bytes, sha256_file, utc8_now_iso


class DownloadError(RuntimeError):
    """A source could not be downloaded after all retries."""


def fetch_bytes(
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float = 60.0,
    retries: int = 5,
    accept_status: tuple[int, ...] = (200, 201),
) -> bytes:
    """GET ``url`` (or POST when ``json_body`` is given) and return the body."""
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if json_body is None:
                response = requests.get(url, headers=headers, timeout=timeout)
            else:
                response = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if response.status_code in accept_status and response.content:
                return response.content
            last_error = DownloadError(f"HTTP {response.status_code} for {url}")
        except requests.RequestException as error:
            last_error = error
        time.sleep(min(4.0 * 2**attempt, 32.0))
    raise DownloadError(f"Could not download {url}: {last_error}")


class Provenance:
    """URL, download time, size, and SHA-256 of every raw snapshot."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            self.entries = json.loads(path.read_text(encoding="utf-8"))["files"]

    def record(
        self,
        file_path: Path,
        *,
        url: str,
        source: str,
        description: str,
        request_body: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        downloaded_at: str | None = None,
    ) -> None:
        rel = file_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        entry: dict[str, Any] = {
            "source": source,
            "description": description,
            "url": url,
            "downloaded_at_utc8": downloaded_at or utc8_now_iso(),
            "bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        }
        if request_body is not None:
            entry["request_body"] = request_body
        entry.update(extra or {})
        self.entries[rel] = entry

    def downloaded_at(self, file_path: Path, **must_match: Any) -> str | None:
        """Time of the earlier download, if the recorded entry still describes the same bytes."""
        entry = self.entries.get(file_path.resolve().relative_to(PROJECT_ROOT).as_posix())
        if entry and all(entry.get(key) == value for key, value in must_match.items()):
            return entry["downloaded_at_utc8"]
        return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0.0",
            "note": (
                "Raw snapshots for Plan 2. Paths are relative to plan_2_tourism_hotel/. "
                "Every file was downloaded from the official source at the URL shown."
            ),
            "files": dict(sorted(self.entries.items())),
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def save_snapshot(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)
