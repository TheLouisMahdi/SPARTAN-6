"""Explicit offline dataset preparation, import and encoding functions."""

import argparse
import csv
import hashlib
import json
import time
import random
import shutil
import sys
import http.client
import os
import re
import tarfile
import urllib.error
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from dataset import read_roi, DatasetError, FIELDS, dataset_lock, summarize, validate_dataset
from encoders import METHODS, pack2
from uuid import uuid4
from collections import Counter
from datetime import datetime, timezone
from capture import ROOT, CODE_ROOT, ASSETS

# Encode Dataset

SPLITS = ("train", "val", "test", "local_holdout")
LABELS = {"OPEN": 0, "CLOSED": 1}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_partitions(data_dir, splits=SPLITS):
    """Fail closed on broken split metadata, before creating any output."""
    data_dir = Path(data_dir).resolve()
    partitions, subjects, frames, identifiers = {}, {}, {}, set()
    for split in splits:
        with (data_dir / "research" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty partition: {split}")
        counts = {label: 0 for label in LABELS}
        for row in rows:
            if row["split"] != split or row["label"] not in LABELS:
                raise ValueError(f"Invalid split or label: {row}")
            if row["sample_id"] in identifiers:
                raise ValueError(f"Repeated sample: {row['sample_id']}")
            identifiers.add(row["sample_id"])
            for mapping, field in ((subjects, "subject_id"), (frames, "frame_group")):
                previous = mapping.setdefault(row[field], split)
                if previous != split:
                    raise ValueError(f"Cross-split {field}: {row[field]}")
            path = (data_dir / row["roi_path"]).resolve()
            if not path.is_relative_to(data_dir) or not path.is_file():
                raise ValueError(f"Missing/unsafe ROI path: {row['roi_path']}")
            counts[row["label"]] += 1
        if not all(counts.values()):
            raise ValueError(f"Both classes required in {split}: {counts}")
        partitions[split] = rows
    return partitions


def build(data_dir, output, methods=None, extra_sources=(), splits=SPLITS):
    methods = METHODS if methods is None else methods
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    partitions = load_partitions(data_dir, splits)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite encoded bundle: {output}")
    stage = output.with_name(output.name + ".partial")
    # An interrupted build remains explicitly incomplete, never a valid bundle.
    stage.mkdir(parents=True, exist_ok=False)
    info = {
        "schema_version": 1, "shape": [16, 32], "dtype": "uint8",
        "input_policy": "Same read-only saved canonical array passed once to every encoder; no recropping, resizing, enhancement, or relabeling.",
        "ordering": "CSV row order, row-major pixels; 2-bit: p0 bits7:6, p1 bits5:4, p2 bits3:2, p3 bits1:0",
        "labels": LABELS, "methods": {name: method.SPEC for name, method in methods.items()},
        "split_info_sha256": sha256(data_dir / "research" / "split_info.json"),
        "source_code_sha256": {"encoders/" + path.name: sha256(path)
                               for path in sorted(Path(__file__).parent.joinpath("encoders").glob("*.py"))},
        "splits": {},
        "timing_policy": "Single offline pass, perf_counter_ns, compute and packing only, excludes PNG I/O; aggregate wall time includes Python overhead. Not capture, sensor, CNN or FPGA latency; no real-time guarantee.",
    }
    for source in extra_sources:
        path = Path(source).resolve()
        info["source_code_sha256"][path.relative_to(Path(__file__).resolve().parent).as_posix()] = sha256(path)
    totals = {name: {"encode_ns": 0, "pack_ns": 0} for name in methods}
    started = time.perf_counter()
    for name in methods:
        (stage / name).mkdir()
    for split, rows in partitions.items():
        handles = {name: (stage / name / f"{split}.bin").open("xb") for name in methods}
        canonical_digest = hashlib.sha256()
        try:
            for index, row in enumerate(rows):
                roi = read_roi(data_dir / row["roi_path"])
                roi.flags.writeable = False
                canonical_digest.update(roi.tobytes())
                for name, method in methods.items():
                    tick = time.perf_counter_ns()
                    codes = method.encode(roi)
                    encoded = time.perf_counter_ns()
                    payload = codes.tobytes() if name == "grayscale8" else pack2(codes)
                    packed = time.perf_counter_ns()
                    totals[name]["encode_ns"] += encoded - tick
                    totals[name]["pack_ns"] += packed - encoded
                    handles[name].write(payload)
                if (index + 1) % 20000 == 0:
                    print(f"{split}: {index + 1}/{len(rows)}", flush=True)
        finally:
            for handle in handles.values():
                handle.close()
        records = {}
        for name, method in methods.items():
            path = stage / name / f"{split}.bin"
            expected = len(rows) * method.SPEC["bytes_per_eye"]
            if path.stat().st_size != expected:
                raise ValueError(f"Incomplete encoded payload: {path}")
            records[name] = {"bytes": expected, "sha256": sha256(path)}
        info["splits"][split] = {
            "count": len(rows), "csv_sha256": sha256(data_dir / "research" / f"{split}.csv"),
            "canonical_sha256": canonical_digest.hexdigest(), "files": records,
        }
        print(f"{split}: {len(rows)} eyes complete", flush=True)
    total = sum(len(rows) for rows in partitions.values())
    info["timing"] = {name: {"eyes": total, **values,
                            "mean_encode_us": values["encode_ns"] / total / 1000,
                            "mean_pack_us": values["pack_ns"] / total / 1000}
                      for name, values in totals.items()}
    info["elapsed_seconds"] = time.perf_counter() - started
    (stage / "bundle.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stage.rename(output)
    print(f"Published: {output}", flush=True)


def encode_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS / "data")
    parser.add_argument("--output", type=Path, default=Path(ASSETS / "data/encoded/v1"))
    args = parser.parse_args()
    try:
        build(args.data_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Encoding failed: {exc}\n")


# Prepare Dataset

def assign_subjects(rows: list[dict], seed: int) -> dict[str, list[str]] | None:
    """Sort before shuffling; keep every session/eye of a subject together."""
    subjects = sorted({row["subject_id"] for row in rows})
    count = len(subjects)
    if count < 3:
        return None
    random.Random(seed).shuffle(subjects)
    train_count = min(count - 2, max(1, round(0.70 * count)))
    val_count = min(count - train_count - 1, max(1, round(0.15 * count)))
    return {
        "train": sorted(subjects[:train_count]),
        "val": sorted(subjects[train_count:train_count + val_count]),
        "test": sorted(subjects[train_count + val_count:]),
    }


def write_splits(root: Path, rows: list[dict], seed: int, replace: bool = False) -> dict | None:
    """Publish all split files together; caller validates and holds dataset lock."""
    assignment = assign_subjects(rows, seed)
    if assignment is None:
        return None
    root = Path(root).resolve()
    destination = root / "splits"
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise DatasetError(f"Split destination must be a real directory: {destination}")
    if destination.exists() and not replace:
        raise DatasetError("Splits already exist. Inspect them first; use --replace-splits explicitly to replace them.")
    info = {
        "schema_version": 1,
        "seed": seed,
        "policy": "Sorted subject IDs, seeded shuffle; all sessions and both eyes stay with subject. "
                  "Rounded 70% train / 15% val, clamped to leave at least one subject in each split; remainder test.",
        "target_subject_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "manifest_sha256": hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest(),
        "path_base": "dataset root (not the splits directory)",
        "subjects": assignment,
        "summary": {},
        "statistical_note": "Only three subjects: statistically limited." if len({r["subject_id"] for r in rows}) == 3
                            else "Subject counts and class balance may limit evaluation; ratios are approximate.",
    }
    staging = root / f".splits-{uuid4().hex}"
    backup = root / f".splits-backup-{uuid4().hex}"
    staging.mkdir()
    try:
        for split, subjects in assignment.items():
            selected = sorted((row for row in rows if row["subject_id"] in subjects), key=lambda r: r["sample_id"])
            info["summary"][split] = summarize(selected)
            with (staging / f"{split}.csv").open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(selected)
        with (staging / "split_info.json").open("x", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return info


def prepare_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-public", action="store_true", help="Validate imported RT-BENE and prepare research splits plus a local holdout.")
    parser.add_argument("--replace-splits", action="store_true", help="Explicitly replace an existing split directory.")
    args = parser.parse_args()
    try:
        if not args.data_dir.is_dir():
            raise DatasetError(f"Dataset directory does not exist: {args.data_dir}")
        with dataset_lock(args.data_dir):
            if args.include_public:

                prepare_research(args.data_dir.resolve(), args.seed, args.replace_splits)
                return 0
            rows = validate_dataset(args.data_dir)
            summary = summarize(rows)
            print("Dataset valid:\n" + json.dumps(summary, indent=2, sort_keys=True))
            if summary["subjects"] < 3:
                print("Fewer than three subjects: independent train/val/test evaluation is not yet possible. "
                      "No splits generated.")
                if (args.data_dir / "splits").exists():
                    print("WARNING: existing splits were left unchanged; they do not describe this validated dataset.")
                return 0
            info = write_splits(args.data_dir, rows, args.seed, args.replace_splits)
            print("Split subjects:\n" + json.dumps(info["subjects"], indent=2))
            print("Actual split counts and distributions:\n" + json.dumps(info["summary"], indent=2, sort_keys=True))
            print(info["statistical_note"])
    except (DatasetError, OSError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


# Import Public Dataset

RECORD_URL = "https://zenodo.org/api/records/3685316"
SOURCE_URL = "https://zenodo.org/records/3685316"
POLICY = "RTBENE_natural_center_60x30_BGR2GRAY_AREA_32x16_v1"
PUBLIC_FIELDS = (
    "sample_id", "subject_id", "source_sample_id", "source_label", "left_label", "right_label",
    "left_path", "right_path", "roi_policy", "source_left_path", "source_right_path",
    "left_sha256", "right_sha256",
)
RESEARCH_FIELDS = (
    "sample_id", "subject_id", "session_id", "frame_group", "eye_side", "label", "roi_path",
    "source_dataset", "roi_policy", "split",
)


def digest_file(path: Path, algorithm="sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, rows, fields):
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_source(downloads: Path) -> dict:
    path = downloads / "record.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(RECORD_URL, timeout=45) as response:
            record = json.load(response)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
    if record["id"] != 3685316 or record["metadata"]["license"]["id"] != "cc-by-nc-sa-4.0":
        raise DatasetError("Unexpected RT-BENE record or source license.")
    return record


def download_file(entry: dict, downloads: Path) -> Path:
    """Cache official files, verify published length and MD5, retry boundedly."""
    path = downloads / entry["key"]
    algorithm, expected = entry["checksum"].split(":", 1)
    if path.exists():
        if path.stat().st_size != entry["size"] or digest_file(path, algorithm) != expected:
            raise DatasetError(f"Existing download failed its official checksum: {path}")
        print(f"Verified cache: {path.name}", flush=True)
        return path
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["size"]:
        raise DatasetError(f"Oversized partial download: {partial}")
    print(f"Downloading {path.name} ({entry['size'] / 1e6:.1f} MB), resume at {offset / 1e6:.1f} MB", flush=True)
    # The connection has actually truncated large responses on this network.
    # Small HTTP ranges allow verified completion without restarting big files.
    while offset < entry["size"]:
        end = min(offset + 8 * 1024 * 1024, entry["size"]) - 1
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(entry["links"]["self"], headers={
                    "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity",
                })
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected_range = f"bytes {offset}-{end}/{entry['size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise DatasetError("Server did not honor the requested download range.")
                    block = response.read(end - offset + 1)
                if len(block) != end - offset + 1:
                    raise DatasetError("Truncated download range.")
                with partial.open("ab") as handle:
                    handle.write(block)
                offset += len(block)
                print(f"  {path.name}: {offset / 1e6:.1f} MB", flush=True)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead, DatasetError) as exc:
                if attempt == 3:
                    raise DatasetError(f"Cannot download {path.name}: {exc}") from exc
                print(f"  Retry {attempt}: {exc}", flush=True)
                time.sleep(2)
    if digest_file(partial, algorithm) != expected:
        raise DatasetError(f"Download checksum mismatch: {partial}")
    partial.rename(path)
    return path


def source_subjects(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 17 or any(len(row) != 6 for row in rows):
        raise DatasetError("Unexpected official subject index.")
    if {int(row[0]) for row in rows} != set(range(17)):
        raise DatasetError("Duplicate or missing official subject ID.")
    return [row for row in rows if row[4] != "discarded"]


def binary_label(value: str) -> str | None:
    # These are publisher-supplied manual labels, never predictions.
    if value == "0.0":
        return "OPEN"
    if value == "1.0":
        return "CLOSED"
    if value == "0.5":
        return None
    raise DatasetError(f"Unexpected RT-BENE label: {value!r}")


def adapt_patch(content: bytes) -> np.ndarray:
    """Source-specific adapter, NOT the accepted Phase 1 corner crop policy."""
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (36, 60, 3) or image.dtype != np.uint8:
        raise DatasetError("Expected an original RT-BENE natural BGR uint8 (36,60,3) patch.")
    # Remove three rows per margin to preserve aspect ratio, without stretching,
    # flipping, enhancement, label-dependent cropping or inferred eye corners.
    gray = cv2.cvtColor(image[3:33, :], cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA))


def import_subject(root: Path, subject: list[str]) -> tuple[list[dict], int]:
    downloads = root / "external" / "rtbene" / "downloads"
    subject_id = f"s{int(subject[0]):03}"
    image_directory = root / "external" / "rtbene" / "images" / subject_id
    image_directory.mkdir(parents=True, exist_ok=True)
    rows, excluded, seen = [], 0, set()
    with (downloads / subject[1]).open(newline="", encoding="utf-8") as handle:
        annotations = list(csv.reader(handle))
    checkpoints = root / "external" / "rtbene" / "subjects"
    checkpoint = checkpoints / f"{subject_id}.csv"
    if checkpoint.exists():
        with checkpoint.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != list(PUBLIC_FIELDS):
                raise DatasetError(f"Invalid import checkpoint: {checkpoint}")
            rows = list(reader)
        if any(row["roi_policy"] != POLICY for row in rows):
            raise DatasetError(f"ROI policy differs in existing import checkpoint: {checkpoint}")
        excluded = sum(binary_label(label) is None for _, label in annotations)
        if len(rows) != len(annotations) - excluded:
            raise DatasetError(f"Incomplete import checkpoint: {checkpoint}")
        print(f"Reusing {subject_id}: {len(rows)} pairs; full validation follows before split publication.", flush=True)
        return rows, excluded
    with tarfile.open(downloads / f"{subject_id}_noglasses_eyes.tar", "r:") as archive:
        members = {entry.name: entry for entry in archive.getmembers() if entry.isfile()}
        if len(members) != sum(entry.isfile() for entry in archive.getmembers()):
            raise DatasetError(f"Duplicate archive member in {subject_id}")
        for annotation in annotations:
            if len(annotation) != 2 or not re.fullmatch(r"left_\d{6}_rgb.png", annotation[0]):
                raise DatasetError(f"Invalid source annotation in {subject_id}: {annotation}")
            filename, source_label = annotation
            if filename in seen:
                raise DatasetError(f"Duplicate source annotation: {subject_id}/{filename}")
            seen.add(filename)
            label = binary_label(source_label)
            if label is None:
                excluded += 1
                continue
            source_id = filename.removeprefix("left_").removesuffix("_rgb.png")
            row = dict(sample_id=f"RTBENE_{subject_id}_{source_id}", subject_id=f"RTBENE_{subject_id}",
                       source_sample_id=source_id, source_label=source_label,
                       left_label=label, right_label=label, roi_policy=POLICY)
            for side, source_folder in (("left", subject[2]), ("right", subject[3])):
                original_path = source_folder + filename.replace("left_", side + "_", 1)
                if original_path not in members:
                    raise DatasetError(f"Missing source eye: {original_path}")
                # Read a named regular member only; never extract archive paths.
                with archive.extractfile(members[original_path]) as handle:
                    roi = adapt_patch(handle.read())
                ok, encoded = cv2.imencode(".png", roi)
                if not ok:
                    raise DatasetError(f"PNG encoding failed: {original_path}")
                content = encoded.tobytes()
                path = image_directory / f"{side}_{source_id}.png"
                expected = hashlib.sha256(content).hexdigest()
                if path.exists():
                    if digest_file(path) != expected:
                        raise DatasetError(f"Refusing to replace differing imported image: {path}")
                else:
                    temporary = path.with_suffix(".png.part")
                    with temporary.open("wb") as handle:
                        handle.write(content)
                    temporary.rename(path)
                row[f"{side}_path"] = path.relative_to(root).as_posix()
                row[f"source_{side}_path"] = original_path
                row[f"{side}_sha256"] = expected
            rows.append(row)
            if len(rows) % 2000 == 0:
                print(f"  {subject_id}: {len(rows)} accepted pairs", flush=True)
    print(f"Imported {subject_id}: {len(rows)} pairs; excluded disagreement: {excluded}", flush=True)
    checkpoints.mkdir(exist_ok=True)
    atomic_csv(checkpoint, rows, PUBLIC_FIELDS)
    return rows, excluded


def validate_public(root: Path) -> list[dict]:
    """Check every imported patch and its association with publisher labels."""
    public = root / "external" / "rtbene"
    downloads = public / "downloads"
    record = json.loads((downloads / "record.json").read_text(encoding="utf-8"))
    if record["id"] != 3685316 or record["metadata"]["license"]["id"] != "cc-by-nc-sa-4.0":
        raise DatasetError("Public source metadata does not identify the licensed RT-BENE release.")
    official = {entry["key"]: entry for entry in record["files"]}
    def verify_annotation(name):
        if name not in official:
            raise DatasetError(f"Missing official checksum for {name}")
        entry = official[name]
        algorithm, expected = entry["checksum"].split(":", 1)
        path = downloads / name
        if path.stat().st_size != entry["size"] or digest_file(path, algorithm) != expected:
            raise DatasetError(f"Publisher annotation checksum failed: {name}")
    verify_annotation("rt_bene_subjects.csv")
    with (public / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != list(PUBLIC_FIELDS):
            raise DatasetError("Unexpected public manifest header.")
        rows = list(reader)
    subjects = source_subjects(downloads / "rt_bene_subjects.csv")
    annotations = {}
    for subject in subjects:
        verify_annotation(subject[1])
        sid = f"RTBENE_s{int(subject[0]):03}"
        with (downloads / subject[1]).open(newline="", encoding="utf-8") as handle:
            annotations[sid] = {name.removeprefix("left_").removesuffix("_rgb.png"): label
                                for name, label in csv.reader(handle) if binary_label(label) is not None}
    seen, paths, errors = set(), set(), []
    for number, row in enumerate(rows, 1):
        try:
            if set(row) != set(PUBLIC_FIELDS) or any(value is None for value in row.values()):
                raise DatasetError("Incomplete public CSV row.")
            subject, frame = row["subject_id"], row["source_sample_id"]
            if row["sample_id"] != f"{subject}_{frame}" or row["sample_id"] in seen:
                raise DatasetError("Duplicate or mismatched public sample ID.")
            seen.add(row["sample_id"])
            if subject not in annotations or annotations[subject].get(frame) != row["source_label"]:
                raise DatasetError("Imported label does not match the publisher annotation.")
            if row["roi_policy"] != POLICY:
                raise DatasetError("Unknown public ROI policy.")
            for side in ("left", "right"):
                if row[f"{side}_label"] != binary_label(row["source_label"]):
                    raise DatasetError("Imported OPEN/CLOSED label mismatch.")
                sid = subject.removeprefix("RTBENE_")
                expected = f"external/rtbene/images/{sid}/{side}_{frame}.png"
                original = f"{sid}_noglasses/natural/{side}/{side}_{frame}_rgb.png"
                if row[f"{side}_path"] != expected or row[f"source_{side}_path"] != original:
                    raise DatasetError("Imported eye path/frame association mismatch.")
                path = root / expected
                if not path.resolve().is_relative_to(root):
                    raise DatasetError("Public image is outside the data directory.")
                read_roi(path)
                if digest_file(path) != row[f"{side}_sha256"]:
                    raise DatasetError(f"Imported PNG checksum mismatch: {expected}")
                if expected in paths:
                    raise DatasetError(f"Reused public image: {expected}")
                paths.add(expected)
        except (DatasetError, OSError) as exc:
            errors.append(f"{row.get('sample_id', 'unknown sample')}: {exc}")
        if number % 10000 == 0:
            print(f"  Validated {number}/{len(rows)} public pairs", flush=True)
    if len(rows) != sum(len(labels) for labels in annotations.values()):
        errors.append("Public manifest does not contain every agreed source annotation.")
    for path in (public / "images").rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in paths:
            errors.append(f"Unreferenced public file: {path.relative_to(root)}")
    if errors:
        raise DatasetError("Public validation failed:\n" + "\n".join(errors))
    return rows


def research_rows(local: list[dict], public: list[dict], seed: int) -> tuple[list[dict], dict]:
    """Public subject-only splits; keep the real webcam person as a local holdout."""
    assignment = assign_subjects(public, seed)
    if assignment is None:
        raise DatasetError("At least three public subjects are required.")
    subject_split = {subject: split for split, subjects in assignment.items() for subject in subjects}
    rows = []
    for pair in public:
        for side in ("left", "right"):
            rows.append(dict(
                sample_id=f"{pair['sample_id']}_{side}", subject_id=pair["subject_id"],
                session_id="RTBENE_noglasses_natural", frame_group=pair["sample_id"], eye_side=side,
                label=pair[f"{side}_label"], roi_path=pair[f"{side}_path"], source_dataset="RTBENE",
                roi_policy=POLICY, split=subject_split[pair["subject_id"]],
            ))
    # Local frame IDs/timestamps remain authoritative in data/manifest.csv.
    # Public sensor timestamps are unavailable and are never fabricated.
    for pair in local:
        for side in ("left", "right"):
            rows.append(dict(
                sample_id=f"LOCAL_{pair['sample_id']}_{side}", subject_id=f"LOCAL_{pair['subject_id']}",
                session_id=pair["session_id"], frame_group=f"LOCAL_{pair['sample_id']}", eye_side=side,
                label=pair[f"{side}_label"], roi_path=pair[f"{side}_path"], source_dataset="LOCAL",
                roi_policy="Phase1_eye_corners_1.6_gray_AREA_32x16", split="local_holdout",
            ))
    return sorted(rows, key=lambda row: row["sample_id"]), assignment


def prepare_research(root: Path, seed: int, replace: bool = False) -> dict:
    destination = root / "research"
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise DatasetError("Research destination must be a real directory.")
    if destination.exists() and not replace:
        raise DatasetError("Research splits already exist; use --prepare-only --replace-splits explicitly to rebuild.")
    local = validate_dataset(root)
    print(f"Local data validated: {len(local)} pairs. Validating public PNGs...", flush=True)
    public = validate_public(root)
    rows, assignment = research_rows(local, public, seed)
    # Exact duplicates crossing subject splits would leak image content even
    # though subject IDs do not overlap. Detect and fail instead of hiding them.
    hash_splits = {}
    for pair in public:
        split = next(split for split, subjects in assignment.items() if pair["subject_id"] in subjects)
        for side in ("left", "right"):
            previous = hash_splits.setdefault(pair[f"{side}_sha256"], split)
            if previous != split:
                raise DatasetError("Identical public PNGs cross splits; resolve the duplicate grouping before evaluation.")
    info = {
        "schema_version": 1, "seed": seed, "source_url": SOURCE_URL,
        "license": "CC-BY-NC-SA-4.0 (RT-BENE source and derivatives)",
        "path_base": "data directory, the parent of research/",
        "policy": "Public IDs sorted, seeded subject-only 70/15/15 assignment. Both eyes and all frames "
                  "stay with the source subject. All local webcam data are a separate local_holdout.",
        "public_subjects": assignment,
        "local_holdout_subjects": sorted({r["subject_id"] for r in rows if r["split"] == "local_holdout"}),
        "local_manifest_sha256": digest_file(root / "manifest.csv"),
        "public_manifest_sha256": digest_file(root / "external" / "rtbene" / "manifest.csv"),
        "roi_policies": sorted({row["roi_policy"] for row in rows}),
        "summary": {},
        "limitations": [
            "Public patches use publisher localization, not the Phase 1 eye-corner crop geometry.",
            "Source consensus labels are inherited; they are not newly verified individual-eye annotations.",
            "Classes are imbalanced; no duplication, resampling or augmentation was performed.",
            "One local subject is a personal webcam holdout, not a population-level webcam benchmark.",
            "These new partitions are not the original RT-BENE paper folds; results are not directly comparable.",
        ],
    }
    stage = root / f".research-{uuid4().hex}"
    backup = root / f".research-backup-{uuid4().hex}"
    stage.mkdir()
    try:
        atomic_csv(stage / "manifest.csv", rows, RESEARCH_FIELDS)
        for split in ("train", "val", "test", "local_holdout"):
            selected = [row for row in rows if row["split"] == split]
            atomic_csv(stage / f"{split}.csv", selected, RESEARCH_FIELDS)
            info["summary"][split] = {
                "subjects": len({row["subject_id"] for row in selected}),
                "frames": len({row["frame_group"] for row in selected}),
                "eye_samples": len(selected), "labels": dict(Counter(row["label"] for row in selected)),
            }
        (stage / "split_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if destination.exists():
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print("Research inputs ready:\n" + json.dumps(info["summary"], indent=2), flush=True)
    return info


def import_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS / "data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace-splits", action="store_true")
    args = parser.parse_args()
    if args.download_only and args.prepare_only:
        parser.error("Choose --download-only or --prepare-only, not both.")
    root = args.data_dir.resolve()
    try:
        with dataset_lock(root):
            if not args.prepare_only:
                downloads = root / "external" / "rtbene" / "downloads"
                downloads.mkdir(parents=True, exist_ok=True)
                record = load_source(downloads)
                files = {entry["key"]: entry for entry in record["files"]}
                index = download_file(files["rt_bene_subjects.csv"], downloads)
                subjects = source_subjects(index)
                rows, excluded = [], 0
                for subject in subjects:
                    subject_id = f"s{int(subject[0]):03}"
                    download_file(files[subject[1]], downloads)
                    download_file(files[f"{subject_id}_noglasses_eyes.tar"], downloads)
                    if not args.download_only:
                        imported, skipped = import_subject(root, subject)
                        rows.extend(imported)
                        excluded += skipped
                if args.download_only:
                    print("Official downloads completed and checksums verified.")
                    return 0
                target = root / "external" / "rtbene"
                atomic_csv(target / "manifest.csv", rows, PUBLIC_FIELDS)
                info = {
                    "source_url": SOURCE_URL, "record_id": 3685316, "license": "CC-BY-NC-SA-4.0",
                    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                    "citation": "K. Cortacero, T. Fischer, Y. Demiris. RT-BENE: A Dataset and Baselines for "
                                "Real-Time Blink Estimation in Natural Environments. ICCV Workshops, 2019.",
                    "download_metadata": "downloads/record.json", "roi_policy": POLICY,
                    "subjects": len(subjects), "pairs": len(rows), "eye_samples": 2 * len(rows),
                    "excluded_disagreement_pairs": excluded, "excluded_subjects": ["s006"],
                    "label_mapping": {"0.0": "OPEN", "1.0": "CLOSED", "0.5": "EXCLUDED"},
                    "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                    "unknown_metadata": ["capture_timestamp_ns", "capture_frame_age_ms", "sensor_exposure_time"],
                    "geometry_note": "Publisher natural eye patches are not Phase 1 corner-derived crops. "
                                     "Use this source-specific adapter consistently for every future encoder.",
                }
                (target / "source_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            prepare_research(root, args.seed, args.replace_splits)
    except (DatasetError, OSError, ValueError, tarfile.TarError, cv2.error) as exc:
        print(f"Public dataset import stopped: {exc}", file=sys.stderr)
        return 1
    return 0
