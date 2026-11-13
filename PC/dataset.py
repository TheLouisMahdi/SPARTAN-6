"""Lossless canonical ROI persistence and offline dataset integrity checks."""

from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import cv2
import numpy as np


FRESHNESS_LIMIT_MS = 1000
LABELS = {"OPEN", "CLOSED"}
FIELDS = (
    "sample_id", "subject_id", "session_id", "frame_id", "capture_timestamp_ns",
    "save_timestamp_utc", "left_label", "right_label", "left_path", "right_path",
    "condition", "frame_age_ms",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DatasetError(ValueError):
    """Invalid data or an unsafe persistence operation."""


def validate_identifier(value: str) -> str:
    """Portable identifiers, excluding path separators and Windows device names."""
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value)
            or value.upper() in reserved):
        raise DatasetError(f"Invalid identifier: {value!r}; use 1-64 letters, digits, '_' or '-'.")
    return value


@contextmanager
def dataset_lock(root: Path):
    """One collector/preparer at a time; refuse stale locks rather than guessing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".dataset.lock"
    try:
        handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise DatasetError(f"Dataset is locked: {lock_path}. Close the other process; "
                           "after a crash, remove this lock only after confirming it has stopped.") from exc
    try:
        with handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            yield
    finally:
        lock_path.unlink()


def write_manifest(path: Path, rows):
    """Commit complete CSV contents atomically; caller holds the dataset lock."""
    temporary = path.with_name(f".manifest-{uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(root: Path) -> list[dict]:
    path = Path(root) / "manifest.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != list(FIELDS):
                raise DatasetError(f"{path}: unexpected CSV header; expected {','.join(FIELDS)}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError(f"Cannot read manifest {path}: {exc}") from exc
    if any(set(row) != set(FIELDS) or any(value is None for value in row.values()) for row in rows):
        raise DatasetError(f"{path}: incomplete or extra CSV fields.")
    return rows


def read_roi(path: Path) -> np.ndarray:
    """Decode unchanged, supporting Unicode Windows paths and rejecting non-PNGs."""
    try:
        content = path.read_bytes()
        if not content.startswith(PNG_SIGNATURE):
            raise DatasetError(f"{path}: not a PNG file.")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error) as exc:
        raise DatasetError(f"Cannot read PNG {path}: {exc}") from exc
    if image is None or image.shape != (16, 32) or image.dtype != np.uint8:
        raise DatasetError(f"{path}: expected a decodable single-channel uint8 PNG of shape (16, 32).")
    return image


def validate_dataset(root: Path) -> list[dict]:
    """Return all rows only if every row/file is valid; never filter bad samples."""
    root = Path(root).resolve()
    rows = load_manifest(root)
    errors = []
    sample_ids, frame_keys, used_paths = set(), set(), set()
    subject_spellings, session_spellings, session_conditions, timelines = {}, {}, {}, {}
    for line, row in enumerate(rows, 2):
        prefix = f"manifest.csv line {line}"
        try:
            for field in ("sample_id", "subject_id", "session_id"):
                validate_identifier(row[field])
            sample_id, subject, session = (row[field] for field in ("sample_id", "subject_id", "session_id"))
            if sample_id in sample_ids:
                raise DatasetError(f"Duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            subject_key, session_key = subject.casefold(), (subject.casefold(), session.casefold())
            if subject_spellings.setdefault(subject_key, subject) != subject:
                raise DatasetError("Subject IDs differ only by letter case.")
            if session_spellings.setdefault(session_key, session) != session:
                raise DatasetError("Session IDs differ only by letter case.")
            if session_conditions.setdefault(session_key, row["condition"]) != row["condition"]:
                raise DatasetError("Conflicting condition metadata for the same session.")
            if len(row["condition"]) > 80 or any(ord(c) < 32 for c in row["condition"]):
                raise DatasetError("Condition must be at most 80 printable characters.")
            frame_id, timestamp = int(row["frame_id"]), int(row["capture_timestamp_ns"])
            if frame_id < 1 or timestamp < 0:
                raise DatasetError("Invalid original frame ID or monotonic capture timestamp.")
            frame_key = (subject_key, session.casefold(), frame_id)
            if frame_key in frame_keys:
                raise DatasetError("Duplicate frame within a subject/session.")
            frame_keys.add(frame_key)
            timelines.setdefault(session_key, []).append((frame_id, timestamp))
            saved_at = datetime.fromisoformat(row["save_timestamp_utc"])
            if saved_at.utcoffset() is None or saved_at.utcoffset().total_seconds() != 0:
                raise DatasetError("Save timestamp must include a UTC offset of zero.")
            age = float(row["frame_age_ms"])
            if not math.isfinite(age) or not 0 <= age <= FRESHNESS_LIMIT_MS:
                raise DatasetError("Recorded frame age exceeds the collection freshness policy.")
            for side in ("left", "right"):
                if row[f"{side}_label"] not in LABELS:
                    raise DatasetError(f"Invalid {side} label; expected OPEN or CLOSED.")
                relative = row[f"{side}_path"]
                expected = f"sessions/{subject}/{session}/{side}_{sample_id}.png"
                if relative != expected:
                    raise DatasetError(f"Unexpected {side} image path: {relative!r}; expected {expected!r}.")
                if relative.casefold() in used_paths:
                    raise DatasetError(f"Image path reused by multiple samples: {relative}")
                used_paths.add(relative.casefold())
                path = root / relative
                if not path.resolve().is_relative_to(root):
                    raise DatasetError(f"Image escapes dataset directory: {relative}")
                read_roi(path)
        except (ValueError, OverflowError) as exc:
            errors.append(f"{prefix}: {exc}")
    for session, frames in timelines.items():
        ordered = sorted(frames)
        if any(current[1] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
            errors.append(f"Session {session}: capture timestamps do not increase with original frame IDs.")
    # A crash can leave images without a manifest row. They are errors, not data.
    for path in (root / "sessions").rglob("*"):
        if path.is_file() and path.suffix.lower() == ".png":
            if path.relative_to(root).as_posix().casefold() not in used_paths:
                errors.append(f"Unreferenced/orphan PNG: {path.relative_to(root)}")
    for path in root.glob(".manifest-*.tmp"):
        errors.append(f"Uncommitted manifest file: {path.name}")
    if errors:
        raise DatasetError("Dataset validation failed:\n" + "\n".join(errors))
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "subjects": len({row["subject_id"] for row in rows}),
        "sessions": len({(row["subject_id"], row["session_id"]) for row in rows}),
        "frames": len(rows),
        "eye_samples": 2 * len(rows),
        "left_labels": dict(Counter(row["left_label"] for row in rows)),
        "right_labels": dict(Counter(row["right_label"] for row in rows)),
        "all_eye_labels": dict(Counter(row[f"{side}_label"] for row in rows for side in ("left", "right"))),
    }


class DatasetSession:
    """One exclusive recording run; saving is synchronous in the main thread."""

    def __init__(self, root: Path, subject_id: str, session_id: str, condition: str = ""):
        self.root = Path(root).resolve()
        self.subject_id = validate_identifier(subject_id)
        self.session_id = validate_identifier(session_id)
        if len(condition) > 80 or any(ord(c) < 32 for c in condition):
            raise DatasetError("Condition must be at most 80 printable characters.")
        self.condition = condition
        self._lock = dataset_lock(self.root)
        self._lock.__enter__()
        self._closed = False
        try:
            manifest = self.root / "manifest.csv"
            if not manifest.exists():
                if any((self.root / "sessions").rglob("*.png")):
                    raise DatasetError("PNG files exist without a manifest; repair the dataset first.")
                write_manifest(manifest, [])
            self.rows = validate_dataset(self.root)
            for directory in (self.root / "sessions").glob("*"):
                if directory.name.casefold() == subject_id.casefold() and directory.name != subject_id:
                    raise DatasetError("Reuse the existing subject ID with exactly the same letter case.")
            self.path = self.root / "sessions" / subject_id / session_id
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # mkdir without exist_ok reserves the run, including an empty run.
            self.path.mkdir()
            self.sample_ids = {row["sample_id"] for row in self.rows}
            self.saved_frames = set()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._lock.__exit__(None, None, None)

    def save_pair(self, record, left_label: str, right_label: str, sample_id: str | None = None) -> str:
        if self._closed:
            raise DatasetError("Session is closed.")
        if record is None:
            raise DatasetError("No current processed frame.")
        if left_label not in LABELS or right_label not in LABELS:
            raise DatasetError("Labels must be OPEN or CLOSED.")
        frame_id, timestamp = record["frame_id"], record["capture_timestamp_ns"]
        if type(frame_id) is not int or frame_id < 1 or type(timestamp) is not int or timestamp < 0:
            raise DatasetError("Invalid original frame metadata.")
        if frame_id in self.saved_frames:
            raise DatasetError("This frame has already been saved in this session.")
        age_ms = (time.monotonic_ns() - timestamp) / 1_000_000
        if not 0 <= age_ms <= FRESHNESS_LIMIT_MS:
            raise DatasetError(f"Frame is stale or has an invalid timestamp ({age_ms:.1f} ms).")
        sample_id = validate_identifier(uuid4().hex if sample_id is None else sample_id)
        if sample_id in self.sample_ids:
            raise DatasetError(f"Duplicate sample ID: {sample_id}")
        payloads = {}
        for side in ("left", "right"):
            roi = record[f"{side}_eye_roi"]
            if (not isinstance(roi, np.ndarray) or roi.shape != (16, 32)
                    or roi.dtype != np.uint8 or not roi.flags.c_contiguous):
                raise DatasetError(f"{side.capitalize()} ROI unavailable or not canonical uint8 (16, 32).")
            ok, encoded = cv2.imencode(".png", roi)
            if not ok:
                raise DatasetError(f"Cannot encode {side} PNG.")
            payloads[side] = encoded.tobytes()
        row = dict(zip(FIELDS, (
            sample_id, self.subject_id, self.session_id, str(frame_id), str(timestamp),
            datetime.now(timezone.utc).isoformat(timespec="microseconds"), left_label, right_label,
            f"sessions/{self.subject_id}/{self.session_id}/left_{sample_id}.png",
            f"sessions/{self.subject_id}/{self.session_id}/right_{sample_id}.png",
            self.condition, f"{age_ms:.6f}",
        )))
        created = []
        try:
            for side, payload in payloads.items():
                path = self.root / row[f"{side}_path"]
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            # Both PNGs are complete before the new row can become visible.
            write_manifest(self.root / "manifest.csv", [*self.rows, row])
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        self.rows.append(row)
        self.sample_ids.add(sample_id)
        self.saved_frames.add(frame_id)
        return sample_id
