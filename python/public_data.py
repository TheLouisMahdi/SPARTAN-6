"""Explicit licensed local sources, canonical eye ROIs and frozen subject splits."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import cv2
import numpy as np
from data_tools import sha256, source_subjects, import_subject, LABELS

DATASETS = {"RTBENE", "MRL", "UTA-RLDD", "DROZY"}
FIELDS = ("sample_id", "subject_id", "session_id", "frame_group", "eye_side", "label", "roi_path",
          "source_dataset", "roi_policy", "split", "canonical_sha256", "label_scope")


def read_registry(path):
    """Read explicit source/rights metadata; reject unapproved, unknown or personal sources."""
    path = Path(path).resolve()
    registry = json.loads(path.read_text(encoding="utf-8-sig"))
    if registry.get("schema_version") != 1 or not registry.get("datasets"):
        raise ValueError("Expected schema_version=1 and a nonempty datasets list")
    seen = set()
    for item in registry["datasets"]:
        name = item["dataset"]
        if name not in DATASETS or name in seen:
            raise ValueError("Use one explicit RTBENE/MRL/UTA-RLDD/DROZY entry per dataset")
        seen.add(name)
        for key in ("version", "source_url", "license", "rights_note", "root"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{name}: missing {key}")
        if item.get("approved_local_use") is not True:
            raise ValueError(f"{name}: confirm lawful local research use in approved_local_use; no access bypass")
        root = Path(item["root"])
        if not root.is_absolute():
            root = path.parent / root
        item["root"] = str(root.resolve())
        if "personal" in str(root).lower():
            raise ValueError("Historical personal data are excluded from this public protocol")
        if not root.is_dir():
            raise FileNotFoundError(f"{name}: obtain authorized source files locally: {root}")
    return registry


def write_csv(path, rows, fields):
    """Write a new UTF-8 CSV, refusing to replace an existing artifact."""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subject_splits(subjects, seed=42):
    """Frozen 60/20/20 subject allocation; sequence/eye records never become split units."""
    subjects = sorted(set(subjects))
    if len(subjects) < 3:
        raise ValueError("At least three public identities are needed for train/val/test")
    random.Random(seed).shuffle(subjects)
    n_test = max(1, round(len(subjects) * 0.2))
    n_val = max(1, round(len(subjects) * 0.2))
    return {subject: ("test" if index < n_test else "val" if index < n_test + n_val else "train")
            for index, subject in enumerate(subjects)}


def validate_rows(rows):
    """Reject subject/sequence/hash leakage, reused sample IDs and inconsistent duplicate labels."""
    identifiers, identities, sessions, frames, hashes = set(), {}, {}, {}, {}
    for row in rows:
        if row["sample_id"] in identifiers or row["label"] not in LABELS or row["label_scope"] != "eye":
            raise ValueError("Duplicate sample ID or invalid eye annotation")
        identifiers.add(row["sample_id"])
        for mapping, key in ((identities, row["subject_id"]),
                             (sessions, (row["subject_id"], row["session_id"])),
                             (frames, row.get("frame_group", row["sample_id"]))):
            if mapping.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Cross-split subject/session leakage")
        digest = row["canonical_sha256"]
        if digest in hashes:
            previous = hashes[digest]
            reason = "inconsistent labels" if previous["label"] != row["label"] else "duplicate canonical frame"
            raise ValueError(f"{reason}: {previous['sample_id']} and {row['sample_id']}; resolve before freezing")
        hashes[digest] = row


def mrl_samples(root, output):
    """MRL official filename fields: subject,index,gender,glasses,eye,reflection,lighting,sensor.

    eye 0=CLOSED,1=OPEN. Deterministic center crop to 2:1, grayscale AREA resize
    to uint8 16x32; this is a documented IR domain adapter, not a landmark crop.
    """
    for path in sorted(Path(root).rglob("*.png")):
        parts = path.stem.split("_")
        if len(parts) != 8 or not parts[0].startswith("s") or parts[4] not in ("0", "1"):
            raise ValueError(f"Invalid official MRL filename: {path.name}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.dtype != np.uint8:
            raise ValueError(f"Unreadable MRL image: {path}")
        height, width = image.shape
        h = min(height, width // 2)
        if h < 1:
            raise ValueError("MRL image is too small")
        y, x = (height - h) // 2, (width - 2 * h) // 2
        roi = cv2.resize(image[y:y + h, x:x + 2 * h], (32, 16), interpolation=cv2.INTER_AREA)
        target = output / "images/MRL" / parts[0] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), roi):
            raise OSError(f"Cannot write canonical ROI: {target}")
        yield dict(sample_id="MRL_" + path.stem, subject_id="MRL_" + parts[0],
                   session_id="MRL_sensor_" + parts[7], frame_group="MRL_" + path.stem, eye_side="unknown",
                   label="OPEN" if parts[4] == "1" else "CLOSED", roi_path=target.relative_to(output).as_posix(),
                   source_dataset="MRL", roi_policy="mrl-center2to1-gray-AREA-v1", label_scope="eye",
                   canonical_sha256=hashlib.sha256(roi.tobytes()).hexdigest(),
                   source_path=str(path), source_sha256=sha256(path), glasses=parts[3], lighting=parts[6], sensor=parts[7])


def rtbene_samples(entry, output):
    """Reuse the existing official RT-BENE patch/label importer; source files are hard-linked locally."""
    root = Path(entry["root"])
    index = root / "rt_bene_subjects.csv"
    subjects = source_subjects(index)
    requested = entry.get("subjects")
    if not requested or not all(isinstance(number, int) for number in requested):
        raise ValueError("RTBENE requires explicit integer subjects, e.g. [0,1,2]; discarded subjects remain excluded")
    selected = [subject for subject in subjects if int(subject[0]) in requested]
    if len(selected) != len(set(requested)):
        raise ValueError("Unknown/discarded/duplicate RTBENE subject requested")
    downloads = output / "external/rtbene/downloads"
    downloads.mkdir(parents=True)
    provenance = {"rt_bene_subjects.csv": sha256(index)}
    for subject in selected:
        sid = f"s{int(subject[0]):03}"
        for name in (subject[1], sid + "_noglasses_eyes.tar"):
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(f"Download official RT-BENE source manually: {source}")
            provenance[name] = sha256(source)
            os.link(source, downloads / name)
        pairs, _ = import_subject(output, subject)
        for pair in pairs:
            for side in ("left", "right"):
                path = output / pair[side + "_path"]
                roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                yield dict(sample_id=pair["sample_id"] + "_" + side, subject_id=pair["subject_id"],
                           session_id="noglasses_natural", frame_group=pair["sample_id"], eye_side=side,
                           label=pair[side + "_label"], roi_path=pair[side + "_path"], source_dataset="RTBENE",
                           roi_policy=pair["roi_policy"], label_scope="eye",
                           canonical_sha256=hashlib.sha256(roi.tobytes()).hexdigest())
    (output / "rtbene_sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def registered_videos(entry):
    """Validate manual video records and independent clip labels, never create eye-state labels."""
    root = Path(entry["root"])
    manifest = root / entry.get("videos_csv", "videos.csv")
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    seen_ids, seen_hashes = set(), set()
    for row in rows:
        for key in ("subject_id", "session_id", "video_id", "path", "sha256", "label_scope", "source_label"):
            if not row.get(key):
                raise ValueError(f"Video manifest missing {key}")
        path = (root / row["path"]).resolve()
        row["sha256"] = row["sha256"].lower()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Missing/unsafe licensed video: {path}")
        if row["label_scope"] != "video":
            raise ValueError("UTA-RLDD/DROZY sleepiness labels apply to videos, not eye frames")
        if entry["dataset"] == "UTA-RLDD" and row["source_label"] not in ("0", "5", "10"):
            raise ValueError("UTA-RLDD clip label must be 0/5/10")
        if entry["dataset"] == "DROZY" and not 1 <= float(row["source_label"]) <= 9:
            raise ValueError("Supply original DROZY KSS 1..9 and annotation provenance")
        if sha256(path) != row["sha256"].lower():
            raise ValueError("Video SHA256 differs from registered source")
        if row["video_id"] in seen_ids or row["sha256"] in seen_hashes:
            raise ValueError("Duplicate video ID/content")
        seen_ids.add(row["video_id"])
        seen_hashes.add(row["sha256"])
        row.update(path=str(path), dataset=entry["dataset"], dataset_version=entry["version"],
                   source_url=entry["source_url"], license=entry["license"], rights_note=entry["rights_note"],
                   annotation_manifest_sha256=sha256(manifest))
    return rows


def prepare(registry_path, output):
    """Lawful local ROIs -> new canonical files, disjoint CSVs and source/split hashes; no private data."""
    registry = read_registry(registry_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, videos = [], []
    external = set()
    for entry in registry["datasets"]:
        if entry["dataset"] in ("UTA-RLDD", "DROZY"):
            videos.extend(registered_videos(entry))
            continue
        entries = list(rtbene_samples(entry, output) if entry["dataset"] == "RTBENE" else mrl_samples(entry["root"], output))
        if not entries:
            raise ValueError(f"No source ROIs found for {entry['dataset']}")
        for row in entries:
            row["subject_id"] = registry.get("identity_map", {}).get(row["subject_id"], row["subject_id"])
            if entry.get("external_test"):
                external.add(row["subject_id"])
        rows.extend(entries)
    allocation = subject_splits([r["subject_id"] for r in rows if r["subject_id"] not in external]) if rows else {}
    for row in rows:
        row["split"] = "external_test" if row["subject_id"] in external else allocation[row["subject_id"]]
    validate_rows(rows)
    research = output / "research"
    research.mkdir()
    splits = ("train", "val", "test") + (("external_test",) if external else ()) if rows else ()
    for split in splits:
        selected = [{key: row[key] for key in FIELDS} for row in rows if row["split"] == split]
        if {r["label"] for r in selected} != set(LABELS):
            raise ValueError(f"Both eye classes required in {split}; acquire more subjects, do not search split seeds")
        write_csv(research / (split + ".csv"), selected, FIELDS)
    if videos:
        if len({row["sha256"] for row in videos}) != len(videos):
            raise ValueError("Duplicate video content across registered datasets")
        allocation_video = subject_splits([f"{v['dataset']}_{v['subject_id']}" for v in videos])
        for video in videos:
            video["split"] = allocation_video[f"{video['dataset']}_{video['subject_id']}"]
        (output / "videos.json").write_text(json.dumps(videos, indent=2) + "\n", encoding="utf-8")
    info = {"schema_version": 1, "public_only": True, "registry_sha256": sha256(registry_path),
            "registry": registry, "split_seed": 42, "allocation": allocation,
            "splits": list(splits), "rows": len(rows),
            "split_sha256": {s: sha256(research / (s + ".csv")) for s in splits},
            "identity_limitation": "Publisher IDs plus explicit identity_map; unreported cross-dataset identities cannot be proven distinct",
            "timestamps": "Precropped eye datasets have no invented video timestamps",
            "domain_limitations": "RT-BENE source crops and MRL IR center crops differ from live landmark ROI"}
    (research / "split_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "source_rows.csv", rows, tuple(dict.fromkeys(k for r in rows for k in r))) if rows else None
    return info
