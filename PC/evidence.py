"""Public evidence commands, reusing the existing encoders, training and replay."""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from capture import ROOT
from data_tools import build, sha256, load_partitions
from encoders import METHODS, aeee2
from encoders.common import validate_roi
from public_data import prepare, read_registry, registered_videos, write_csv, subject_splits

PROTOCOL = ROOT / "RESEARCH/protocol.json"


class PublicAblation:
    """Explicit 2-bit controls of frozen AEEE v0.1; no data-driven parameters."""
    def __init__(self, component):
        from research import AEEEVariant
        self.component = component
        self.base = None if component in ("strong_split", "polarity_order") else AEEEVariant(component)
        self.SPEC = dict(self.base.SPEC if self.base else aeee2.SPEC)
        self.SPEC.update(version="aeee2-public-control-v1-" + component, removed_or_changed=component,
                         control_policy="No data/labels used; new model per representation; exact 128-byte payload")
        if component == "strong_split":
            self.SPEC["symbols"] = {"0": "not retained", "1": "all retained edges", "2": "unused", "3": "unused"}
        elif component == "polarity_order":
            self.SPEC["symbols"] = {"0": "not retained", "1": "weak retained", "2": "strong Gy<0", "3": "strong Gy>0"}

    def encode(self, roi):
        """uint8 16x32 -> uint8 0..3 codes, preserving validation and payload size."""
        validate_roi(roi)
        if self.base is not None:
            return self.base.encode(roi)
        codes = aeee2.encode(roi)
        if self.component == "strong_split":
            return (codes > 0).astype(np.uint8)
        return np.array([0, 1, 3, 2], np.uint8)[codes]


def encoders(ablations=False):
    """Active two-bit experiments, optionally frozen AEEE constituent controls."""
    result = {name: METHODS[name] for name in ("aeee2", "grayscale2")}
    if ablations:
        protocol = json.loads(PROTOCOL.read_text())
        result.update({"aeee2_no_" + name: PublicAblation(name) for name in protocol["ablations"]})
        result["aeee2_swap_polarity"] = PublicAblation("polarity_order")
    return result


def public_splits(data_dir):
    """Require a prepared public-only split manifest; reject historical local/personal holdouts."""
    info = json.loads((Path(data_dir) / "research/split_info.json").read_text(encoding="utf-8"))
    if info.get("public_only") is not True or info.get("split_seed") != 42:
        raise ValueError("Requires new frozen public subject splits, not historical/personal data")
    if not {"train", "val", "test"}.issubset(info["splits"]):
        raise ValueError("NOT_RUN_DATA_UNAVAILABLE: prepared eye-state train/val/test data required")
    for name, digest in info["split_sha256"].items():
        if sha256(Path(data_dir) / "research" / (name + ".csv")) != digest:
            raise ValueError("Frozen split changed")
    return tuple(info["splits"])


def fit_public(args):
    """Delegate fitting/evaluation to research.run with fixed budget and no local_holdout."""
    from research import run
    protocol = json.loads(PROTOCOL.read_text())
    methods = encoders(args.ablations)
    options = SimpleNamespace(data_dir=args.data_dir, encoded=args.encoded, output=args.output,
        seeds=protocol["seeds"], methods=list(methods), epochs=protocol["epochs"],
        batch_size=protocol["batch_size"], learning_rate=protocol["learning_rate"], device="cpu",
        stage="fit" if args.action == "fit" else "evaluate", resume=args.action == "evaluate",
        splits=public_splits(args.data_dir), public_only=True, protocol_sha256=sha256(PROTOCOL))
    run(options, methods)
    if args.action == "evaluate":
        comparison(args.output, args.encoded, args.data_dir)


def matrix_metrics(matrix):
    """Confusion matrix with actual rows/predicted columns -> class/macro metrics, including zero support."""
    matrix = np.asarray(matrix, np.float64)
    precision = np.divide(matrix.diagonal(), matrix.sum(axis=0), out=np.zeros(len(matrix)), where=matrix.sum(axis=0) != 0)
    recall = np.divide(matrix.diagonal(), matrix.sum(axis=1), out=np.zeros(len(matrix)), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(matrix)), where=precision + recall != 0)
    return {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "f1_closed": float(f1[1]), "closed_recall": float(recall[1])}


def paired_bootstrap(aeee, baseline, resamples=2000, seed=611):
    """Matched [seed,subject,2,2] confusion matrices -> pooled macro-F1 delta subject-bootstrap CI.

    Seeds are averaged, not selected or treated as independent subjects. Matched
    resampled subject indices are shared across methods and seeds.
    """
    aeee, baseline = np.asarray(aeee), np.asarray(baseline)
    if aeee.shape != baseline.shape or aeee.ndim != 4 or aeee.shape[2:] != (2, 2) or aeee.shape[1] < 2:
        raise ValueError("Need paired seed,subject,2,2 matrices and >=2 independent subjects")
    if resamples < 1000:
        raise ValueError("At least 1000 subject bootstrap resamples are required")
    def delta(indices):
        return float(np.mean([matrix_metrics(x[indices].sum(axis=0))["macro_f1"] -
                              matrix_metrics(y[indices].sum(axis=0))["macro_f1"] for x, y in zip(aeee, baseline)]))
    rng = np.random.default_rng(seed)
    n = aeee.shape[1]
    effects = [delta(rng.integers(0, n, n)) for _ in range(resamples)]
    low, high = np.quantile(effects, [0.025, 0.975])
    return {"delta_macro_f1": delta(np.arange(n)), "ci95_low": float(low), "ci95_high": float(high),
            "n_subjects": n, "n_seeds": aeee.shape[0], "resamples": resamples,
            "unit": "subject", "seed": seed, "small_sample_warning": n < 10}


def comparison(run_dir, bundle_dir=None, data_dir=None):
    """New per-person/comparison tables and honest paired outcome from actual frozen evaluations only."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if not config.get("public_only") or config["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("Comparison requires this frozen public protocol")
    protocol = json.loads(PROTOCOL.read_text())
    bundle = json.loads((Path(bundle_dir) / "bundle.json").read_text()) if bundle_dir else None
    if bundle_dir and sha256(Path(bundle_dir) / "bundle.json") != config["bundle_sha256"]:
        raise ValueError("Comparison bundle differs from frozen fit")
    partitions = load_partitions(data_dir, public_splits(data_dir)) if data_dir else None
    if config["seeds"] != protocol["seeds"]:
        raise ValueError("Report all five frozen seeds, never best-seed selection")
    records, subjects, evaluations = [], [], {}
    for method in config["methods"]:
        for seed in config["seeds"]:
            evaluation = json.loads((run_dir / method / f"seed_{seed}/evaluation.json").read_text())
            evaluations[method, seed] = evaluation
            for split, report in evaluation.items():
                metric = report["int8"]
                records.append(dict(encoding=method, split=split, seed=seed, precision="int8", n_eyes=metric["count"],
                    dataset="+".join(sorted({r["source_dataset"] for r in partitions[split]})) if partitions else "declared public sources",
                    n_subjects=len(report["per_subject"]), size_bytes_eye=512 if method == "grayscale8" else 128,
                    avg_encode_ms=bundle["timing"][method]["mean_encode_us"] / 1000 if bundle else None,
                    macro_f1=metric["macro_f1"], balanced_accuracy=metric["balanced_accuracy"],
                    f1_closed=metric["per_class"]["CLOSED"]["f1"], closed_recall=metric["per_class"]["CLOSED"]["recall"],
                    status="PASS", provenance_sha256=sha256(run_dir / "config.json")))
                for person, result in report["per_subject"].items():
                    subjects.append(dict(encoding=method, seed=seed, split=split, subject_id=person,
                                         **matrix_metrics(result["int8"]["confusion_matrix"])))
    write_csv(run_dir / "public_comparison.csv", records, tuple(records[0]))
    write_csv(run_dir / "metrics_per_subject.csv", subjects, tuple(subjects[0]))
    outcomes = {}
    for split in (name for name in config["splits"] if name not in ("train", "val")):
        ids = sorted(evaluations["aeee2", config["seeds"][0]][split]["per_subject"])
        paired = []
        for method in ("aeee2", "grayscale2"):
            paired.append(np.array([[evaluations[method, seed][split]["per_subject"][person]["int8"]["confusion_matrix"]
                                     for person in ids] for seed in config["seeds"]]))
        outcomes[split] = paired_bootstrap(*paired, **{k: protocol["bootstrap"][k] for k in ("resamples", "seed")}) if len(ids) >= 2 else {
            "status": "NOT_RUN_DATA_UNAVAILABLE", "reason": "At least two independent test subjects required for interval"}
    (run_dir / "paired_results.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    lines = ["Public evidence only. No novelty, superiority or clinical claim is assumed."]
    for split, result in outcomes.items():
        if "delta_macro_f1" in result:
            outcome = "interval includes zero" if result["ci95_low"] <= 0 <= result["ci95_high"] else (
                "interval favors AEEE" if result["ci95_low"] > 0 else "interval favors grayscale2")
            lines.append(f"{split}: delta={result['delta_macro_f1']:.6f}, CI=[{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]; {outcome}.")
        else:
            lines.append(f"{split}: {result['reason']}")
    (run_dir / "conclusions.txt").write_text("\n".join(lines) + "\n")
    return outcomes


def replay_registered(args):
    """Validate an authorized video then call the existing headless replay with source PTS."""
    from app import replay_cli
    registry = read_registry(args.registry)
    records = [row for entry in registry["datasets"] if entry["dataset"] in ("UTA-RLDD", "DROZY")
               for row in registered_videos(entry)]
    if len({row["sha256"] for row in records}) != len(records):
        raise ValueError("Duplicate registered video content")
    matches = [record for record in records if record["video_id"] == args.video_id]
    if len(matches) != 1:
        raise ValueError("video_id must identify exactly one registered lawful video")
    record = matches[0]
    allocation = subject_splits([row["dataset"] + "_" + row["subject_id"] for row in records])
    record["split"] = allocation[record["dataset"] + "_" + record["subject_id"]]
    command = [record["path"], "--output", str(args.output), "--backend", args.backend, "--timebase", "pts"]
    if args.backend != "pretrained":
        if args.run_dir is None or args.bundle is None:
            raise ValueError("CNN research replay requires explicit public --run-dir and --bundle")
        config = json.loads((args.run_dir / "config.json").read_text())
        if config.get("public_only") is not True:
            raise ValueError("Historical personal models are excluded from public video results")
        command += ["--run-dir", str(args.run_dir), "--bundle", str(args.bundle), "--method", args.method, "--seed", str(args.seed)]
    result = replay_cli(command)
    if args.output.exists():
        (args.output / "source_video.json").write_text(json.dumps(dict(record, registry_sha256=sha256(args.registry),
            label_warning="clip sleepiness label, never eye-frame ground truth"), indent=2) + "\n")
    return result


def perturbation_report(data_dir, run_dir, bundle, output):
    """Frozen public fits -> heldout-only fixed photometric controls, never retraining or selecting on test."""
    from app import EyePredictor
    from dataset import read_roi
    from integer_model import infer
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if not config.get("public_only") or config["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("Requires a frozen public fit, not a personal checkpoint")
    if not (run_dir / "selection_frozen.json").is_file():
        raise ValueError("Finish all train/validation model selection before heldout perturbations")
    partitions = load_partitions(data_dir, public_splits(data_dir))
    protocol = json.loads(PROTOCOL.read_text())["perturbations"]
    records = []
    for split, rows in partitions.items():
        if split in ("train", "val"):
            continue
        pixels = np.stack([read_roi(Path(data_dir) / row["roi_path"]) for row in rows])
        actual = np.array([int(row["label"] == "CLOSED") for row in rows])
        for condition in ("original", "dark", "bright", "contrast", "blur", "occlusion"):
            changed = pixels.copy()
            if condition in ("dark", "bright"):
                changed = np.clip(changed.astype(np.float32) * protocol[condition], 0, 255).astype(np.uint8)
            elif condition == "contrast":
                changed = np.clip((changed.astype(np.float32) - 127.5) * protocol[condition] + 127.5, 0, 255).astype(np.uint8)
            elif condition == "blur":
                changed = np.stack([cv2.GaussianBlur(roi, tuple(protocol["blur_kernel"]), 0) for roi in pixels])
            elif condition == "occlusion":
                changed[:, :4] = 0
            for method, encoder in METHODS.items():
                encoded = np.stack([encoder.encode(roi) for roi in changed])[:, None]
                if method != "grayscale8":
                    encoded *= np.uint8(85)
                for seed in config["seeds"]:
                    predictor = EyePredictor(method, seed, run_dir, bundle)
                    predicted = np.concatenate([infer(encoded[i:i + 256], predictor.params) for i in range(0, len(encoded), 256)]).argmax(axis=1)
                    for person in sorted({row["subject_id"] for row in rows}):
                        mask = np.array([row["subject_id"] == person for row in rows])
                        matrix = np.bincount(actual[mask] * 2 + predicted[mask], minlength=4).reshape(2, 2)
                        records.append(dict(encoding=method, seed=seed, split=split, condition=condition,
                            subject_id=person, n_eyes=int(mask.sum()), status="PASS", **matrix_metrics(matrix)))
    write_csv(output, records, tuple(records[0]))
    return records


def clip_features(replays, output):
    """Registered replay summaries -> clip-level UTA features/labels; invalid coverage is excluded explicitly."""
    records, excluded = [], []
    for directory in replays:
        directory = Path(directory)
        source = json.loads((directory / "source_video.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        ocular = json.loads((directory / "ocular_metrics.json").read_text())["summary"]
        if source["dataset"] != "UTA-RLDD" or source["label_scope"] != "video" or source["source_label"] not in ("0", "5", "10"):
            raise ValueError("Three-class sleepiness requires independently registered UTA-RLDD clip labels")
        if summary["status"] != "complete" or summary["video_sha256"] != source["sha256"]:
            raise ValueError("Replay is incomplete or does not match registered video")
        eyes = ocular["eyes"]
        if any(eyes.get(side, {}).get("status") != "descriptive_only" for side in ("left", "right")):
            excluded.append(dict(video_id=source["video_id"], reason="low coverage or <30s observed; no imputation"))
            continue
        row = dict(dataset="UTA-RLDD", subject_id=source["subject_id"], video_id=source["video_id"],
                   split=source["split"], label_scope="video", source_label=source["source_label"],
                   source_sha256=source["sha256"], annotation_sha256=source["annotation_manifest_sha256"],
                   replay_summary_sha256=sha256(directory / "summary.json"))
        for side in ("left", "right"):
            row[side + "_closed_fraction"] = eyes[side]["closed_fraction"]
            row[side + "_blink_rate"] = eyes[side]["blink_rate_per_min"]
        records.append(row)
    if not records:
        raise ValueError("NOT_RUN_DATA_UNAVAILABLE: no lawful clips with sufficient valid duration/coverage")
    write_csv(output, records, tuple(records[0]))
    Path(output).with_suffix(".excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    return records


def sleepiness_baseline(features, output):
    """Train-only standardized nearest-centroid clip classifier; independent test labels never fit parameters."""
    with Path(features).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    subjects, videos, digests = {}, set(), set()
    for row in rows:
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Expected frozen train/val/test split")
        if row["label_scope"] != "video" or row["dataset"] != "UTA-RLDD":
            raise ValueError("Use independent UTA-RLDD clip labels, not CNN eye predictions as labels")
        if subjects.setdefault(row["subject_id"], row["split"]) != row["split"]:
            raise ValueError("Cross-split sleepiness subject")
        if row["video_id"] in videos or row["source_sha256"] in digests:
            raise ValueError("Duplicate video content/ID")
        videos.add(row["video_id"])
        digests.add(row["source_sha256"])
    fields = [side + suffix for side in ("left", "right") for suffix in ("_closed_fraction", "_blink_rate")]
    x = np.array([[float(row[f]) for f in fields] for row in rows])
    y = np.array([("0", "5", "10").index(row["source_label"]) for row in rows])
    if not np.isfinite(x).all():
        raise ValueError("Features must be observed finite values; no silent missing-value imputation")
    train = np.array([row["split"] == "train" for row in rows])
    if set(y[train]) != {0, 1, 2}:
        raise ValueError("All three independent clip classes required in TRAIN")
    mean, std = x[train].mean(axis=0), x[train].std(axis=0)
    std[std == 0] = 1
    scaled = (x - mean) / std
    centroids = np.stack([scaled[train & (y == label)].mean(axis=0) for label in range(3)])
    predicted = ((scaled[:, None] - centroids[None]) ** 2).sum(axis=2).argmin(axis=1)
    result = {"status": "PASS", "scope": "clip-level self-report classification, not medical diagnosis",
              "model": "fixed train-standardized nearest centroid, no tuning", "features": fields,
              "train_mean": mean.tolist(), "train_std": std.tolist(), "centroids": centroids.tolist(),
              "source_sha256": sha256(features), "classes": [0, 5, 10], "aggregation": "whole-clip valid-time indicators",
              "splits": {}}
    for split in ("train", "val", "test"):
        mask = np.array([row["split"] == split for row in rows])
        if not mask.any():
            result["splits"][split] = {"status": "NOT_RUN_DATA_UNAVAILABLE"}
            continue
        matrix = np.bincount(y[mask] * 3 + predicted[mask], minlength=9).reshape(3, 3)
        metric = matrix_metrics(matrix)
        result["splits"][split] = dict(status="PASS", n_clips=int(mask.sum()), confusion_matrix=matrix.tolist(),
            macro_f1=metric["macro_f1"], balanced_accuracy=metric["balanced_accuracy"],
            n_subjects=len({rows[i]["subject_id"] for i in np.flatnonzero(mask)}))
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def classify_clip(row, artifact=None):
    """Apply an evaluated whole-clip baseline only to registered whole-clip features."""
    unknown = dict(level="UNKNOWN", scope="whole-clip research classification")
    if artifact is None or not Path(artifact).is_file():
        return dict(unknown, reason="Missing fatigue classifier artifact")
    model = json.loads(Path(artifact).read_text(encoding="utf-8"))
    if (model.get("status") != "PASS" or model.get("classes") != [0, 5, 10]
            or model.get("aggregation") != "whole-clip valid-time indicators"
            or any(model.get("splits", {}).get(s, {}).get("status") != "PASS" for s in ("val", "test"))):
        return dict(unknown, reason="Whole-clip validation/test evidence unavailable")
    if row.get("dataset") != "UTA-RLDD" or row.get("label_scope") != "video" or not all(
            row.get(k) for k in ("subject_id", "source_sha256", "annotation_sha256", "replay_summary_sha256")):
        return dict(unknown, reason="Requires independently registered whole-clip features")
    fields = [s + suffix for s in ("left", "right") for suffix in ("_closed_fraction", "_blink_rate")]
    if model.get("features") != fields:
        return dict(unknown, reason="Feature contract mismatch")
    x = np.array([float(row[f]) for f in fields])
    mean, std, centroids = (np.asarray(model[k], dtype=float) for k in ("train_mean", "train_std", "centroids"))
    if (mean.shape != (4,) or std.shape != (4,) or centroids.shape != (3, 4)
            or not all(np.isfinite(a).all() for a in (x, mean, std, centroids)) or np.any(std <= 0)):
        raise ValueError("Invalid fitted fatigue parameters or observed features")
    predicted = int((((x-mean)/std-centroids)**2).sum(axis=1).argmin())
    return dict(level=("ALERT", "LOW_VIGILANCE", "DROWSY")[predicted], scope=unknown["scope"],
                artifact_sha256=sha256(artifact), validation=model["splits"],
                limitation="Clip-level research only; not validated for live or 30/60-second prediction")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--registry", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("encode", "fit", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data-dir", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--ablations", action="store_true")
        if name != "encode":
            cmd.add_argument("--encoded", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--registry", required=True, type=Path)
    replay.add_argument("--video-id", required=True)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--backend", choices=("pretrained", "int8", "fpga"), default="pretrained")
    replay.add_argument("--run-dir", type=Path)
    replay.add_argument("--bundle", type=Path)
    replay.add_argument("--method", choices=tuple(METHODS), default="aeee2")
    replay.add_argument("--seed", choices=(42, 43, 44, 45, 46), type=int, default=42)
    ocular = sub.add_parser("ocular")
    ocular.add_argument("--frames", type=Path, required=True)
    ocular.add_argument("--output", type=Path, required=True)
    ocular.add_argument("--aperture-provenance", type=Path,
                        help="External calibration provenance JSON; raw EAR/binary CNN states do not qualify")
    perturb = sub.add_parser("perturb")
    perturb.add_argument("--data-dir", type=Path, required=True)
    perturb.add_argument("--run-dir", type=Path, required=True)
    perturb.add_argument("--bundle", type=Path, required=True)
    perturb.add_argument("--output", type=Path, required=True)
    feature = sub.add_parser("features")
    feature.add_argument("--replays", nargs="+", type=Path, required=True)
    feature.add_argument("--output", type=Path, required=True)
    sleep = sub.add_parser("sleepiness")
    sleep.add_argument("--features", type=Path, required=True)
    sleep.add_argument("--output", type=Path, required=True)
    classify = sub.add_parser("classify-clip")
    classify.add_argument("--features", type=Path, required=True)
    classify.add_argument("--model", type=Path)
    classify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "classify-clip":
        with args.features.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        results = [dict(video_id=row.get("video_id"), classification=classify_clip(row, args.model)) for row in rows]
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
        return 0
    if args.action == "prepare":
        print(json.dumps(prepare(args.registry, args.output), indent=2))
    elif args.action == "encode":
        build(args.data_dir, args.output, encoders(args.ablations),
              extra_sources=(ROOT / "PC/evidence.py", ROOT / "PC/research.py"), splits=public_splits(args.data_dir))
    elif args.action in ("fit", "evaluate"):
        fit_public(args)
    elif args.action == "replay":
        return replay_registered(args)
    elif args.action == "perturb":
        perturbation_report(args.data_dir, args.run_dir, args.bundle, args.output)
    elif args.action == "features":
        clip_features(args.replays, args.output)
    elif args.action == "sleepiness":
        sleepiness_baseline(args.features, args.output)
    else:
        from ocular_metrics import export_metrics
        provenance = json.loads(args.aperture_provenance.read_text()) if args.aperture_provenance else None
        export_metrics(args.frames, args.output, aperture_provenance=provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
