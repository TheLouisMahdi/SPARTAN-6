"""Explicit offline fitting and AEEE component ablation; never trains on import."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import os
import argparse
import csv
import hashlib
import json
import platform
import random
import time
import numpy as np
import torch
import integer_model
from pathlib import Path
from data_tools import LABELS, SPLITS, load_partitions, sha256, build
from encoders import METHODS, aeee2
from research_model import ARCHITECTURE, TinyEye
from collections import Counter
from dataset import read_roi, validate_dataset
from app import EyePredictor, ROOT
from encoders.common import sobel_components, validate_roi
from capture import ROOT, CODE_ROOT, ASSETS

# Train Research

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")





def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics(actual, predicted):
    matrix = np.bincount(np.asarray(actual, np.int64) * 2 + predicted, minlength=4).reshape(2, 2)
    recall = np.divide(matrix.diagonal(), matrix.sum(axis=1), out=np.zeros(2), where=matrix.sum(axis=1) != 0)
    precision = np.divide(matrix.diagonal(), matrix.sum(axis=0), out=np.zeros(2), where=matrix.sum(axis=0) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2), where=precision + recall != 0)
    return {"count": int(matrix.sum()), "accuracy": float(matrix.trace() / matrix.sum()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "confusion_matrix": matrix.tolist(), "matrix_order": ["OPEN", "CLOSED"],
            "per_class": {label: {"precision": float(precision[index]), "recall": float(recall[index]),
                                   "f1": float(f1[index]), "support": int(matrix[index].sum())}
                          for label, index in LABELS.items()}}


def load_pixels(bundle_dir, method, split, count):
    payload = np.memmap(bundle_dir / method / f"{split}.bin", mode="r", dtype=np.uint8)
    if method == "grayscale8":
        return np.array(payload).reshape(count, 1, 16, 32)
    values = np.empty((len(payload), 4), np.uint8)
    for index, shift in enumerate((6, 4, 2, 0)):
        values[:, index] = (payload >> shift) & 3
    return (values.reshape(count, 1, 16, 32) * 85).astype(np.uint8)


def targets(rows):
    return np.array([LABELS[row["label"]] for row in rows], np.int64)


def tensor_pixels(pixels, device):
    return torch.from_numpy(pixels).to(device=device, dtype=torch.float32).div_(255)


def float_logits(model, x, batch_size):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(x), batch_size):
            outputs.append(model(x[offset:offset + batch_size]).cpu().numpy())
    return np.concatenate(outputs)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_one(directory, method, seed, pixels, partitions, args, device, encoder=None):
    encoder = METHODS[method] if encoder is None else encoder
    if directory.exists():
        if args.resume and (directory / "fit.json").is_file():
            print(f"Reuse completed fit: {method} seed={seed}", flush=True)
            return
        raise FileExistsError(f"Incomplete/existing fit, use a new output directory: {directory}")
    stage = directory.with_name(directory.name + ".partial")
    stage.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyEye().to(device)
    initial_hash = state_hash(model)
    y = targets(partitions["train"])
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (2 * counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    x_train, x_val = (tensor_pixels(pixels[split], device) for split in ("train", "val"))
    y_train = torch.from_numpy(y).to(device)
    y_val = targets(partitions["val"])
    history, best_score, best_epoch = [], -1.0, None
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.from_numpy(rng.permutation(len(y))).to(device)
        losses = []
        for offset in range(0, len(y), args.batch_size):
            indices = permutation[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        train_loss = float(torch.stack(losses).mean())
        logits = float_logits(model, x_val, args.batch_size)
        val = metrics(y_val, logits.argmax(axis=1))
        history.append({"epoch": epoch, "mean_batch_weighted_train_loss": train_loss, "validation": val})
        if val["macro_f1"] > best_score:
            best_score, best_epoch = val["macro_f1"], epoch
            torch.save({"state_dict": model.state_dict(), "method": method,
                        "encoder_version": encoder.SPEC["version"], "seed": seed,
                        "epoch": epoch, "architecture": ARCHITECTURE}, stage / "float32.pt")
        print(f"{method} seed={seed} epoch={epoch}/{args.epochs} loss={train_loss:.4f} val_macro_f1={val['macro_f1']:.4f}", flush=True)
    checkpoint = torch.load(stage / "float32.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    # Shared index policy per seed. This set is sampled only from TRAIN, without labels.
    calibration_indices = np.sort(np.random.default_rng(seed).choice(len(y), min(1024, len(y)), replace=False))
    calibration = x_train[torch.from_numpy(calibration_indices).to(device)]
    params, quantization = integer_model.make_parameters(model, calibration, args.batch_size)
    integer_model.save_parameters(stage, params, quantization)
    np.save(stage / "calibration_train_indices.npy", calibration_indices, allow_pickle=False)
    info = {"method": method, "seed": seed, "initial_state_sha256": initial_hash,
            "selected_epoch": best_epoch, "selected_validation_macro_f1": best_score,
            "selection": "Maximum validation macro F1, earliest epoch on a tie; no threshold search",
            "train_class_counts_OPEN_CLOSED": counts.tolist(), "class_weights_OPEN_CLOSED": weights.tolist(),
            "history": history, "fit_and_calibration_seconds": time.perf_counter() - start,
            "float32_sha256": sha256(stage / "float32.pt"),
            "int8_sha256": sha256(stage / "int8_parameters.npz")}
    write_json(stage / "fit.json", info)
    stage.rename(directory)


def evaluate_one(directory, method, pixels, partitions, args, device):
    if args.resume and (directory / "evaluation.json").is_file():
        return json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(directory / "float32.pt", map_location=device, weights_only=True)
    model = TinyEye().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    with np.load(directory / "int8_parameters.npz", allow_pickle=False) as archive:
        params = {key: archive[key] for key in archive.files}
    report = {}
    for split in getattr(args, "evaluation_splits", ("val", "test", "local_holdout")):
        rows, source = partitions[split], pixels[split]
        actual = targets(rows)
        x = tensor_pixels(source, device)
        float_scores = float_logits(model, x, args.batch_size)
        del x
        tick = time.perf_counter()
        integer_scores = np.concatenate([integer_model.infer(source[offset:offset + 256], params)
                                         for offset in range(0, len(source), 256)])
        integer_seconds = time.perf_counter() - tick
        fp_pred, int_pred = float_scores.argmax(axis=1), integer_scores.argmax(axis=1)
        result = {"float32": metrics(actual, fp_pred), "int8": metrics(actual, int_pred),
                  "float_int8_prediction_agreement": float(np.mean(fp_pred == int_pred)),
                  "int8_reference_seconds": integer_seconds,
                  "int8_timing_scope": "Batched NumPy integer forward pass plus concatenation, excludes encoding and I/O; CPU reference throughput, not FPGA or single-eye latency",
                  "majority_OPEN": metrics(actual, np.zeros_like(actual)),
                  "per_subject": {}}
        for subject in sorted({row["subject_id"] for row in rows}):
            mask = np.array([row["subject_id"] == subject for row in rows])
            result["per_subject"][subject] = {"float32": metrics(actual[mask], fp_pred[mask]),
                                               "int8": metrics(actual[mask], int_pred[mask])}
        np.savez_compressed(directory / f"{split}_predictions.npz", actual=actual,
                            float_logits=float_scores, int32_logits=integer_scores)
        report[split] = result
        print(f"EVAL {method} seed={checkpoint['seed']} {split}: FP32 F1={result['float32']['macro_f1']:.4f}, INT8 F1={result['int8']['macro_f1']:.4f}", flush=True)
    write_json(directory / "evaluation.json", report)
    return report


def run(args, encoders=None):
    encoders = METHODS if encoders is None else encoders
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require positive epochs/batch/lr and distinct seeds")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    active_splits = tuple(getattr(args, "splits", SPLITS))
    evaluation_splits = tuple(split for split in active_splits if split != "train")
    args.evaluation_splits = evaluation_splits
    partitions = load_partitions(args.data_dir, active_splits)
    bundle = json.loads((args.encoded / "bundle.json").read_text(encoding="utf-8"))
    for split in active_splits:
        record = bundle["splits"][split]
        if record["count"] != len(partitions[split]) or record["csv_sha256"] != sha256(args.data_dir / "research" / f"{split}.csv"):
            raise ValueError(f"Encoded data does not match frozen split: {split}")
        for method in args.methods:
            path = args.encoded / method / f"{split}.bin"
            if sha256(path) != record["files"][method]["sha256"] or path.stat().st_size != len(partitions[split]) * encoders[method].SPEC["bytes_per_eye"]:
                raise ValueError(f"Encoded payload failed verification: {path}")
    if bundle["methods"] != {name: method.SPEC for name, method in encoders.items()}:
        raise ValueError("Encoder definitions differ from encoded bundle")
    for name, digest in bundle["source_code_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"Encoder source differs from encoded bundle: {name}")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Methods must be distinct")
    source_paths = [Path(__file__).with_name(name) for name in
                    ("research.py", "research_model.py", "integer_model.py", "data_tools.py")]
    config = {"schema_version": 1, "architecture": ARCHITECTURE, "seeds": args.seeds, "methods": args.methods,
              "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "Adam",
              "learning_rate": args.learning_rate, "weight_decay": 0, "class_weighting": "N/(2*train_class_count)",
              "input_mapping": "grayscale8: I/255; other methods: ordinal C/3 (integer input 85*C). AEEE polarity categories are a fixed ordinal experimental mapping.",
              "calibration": "1024 train-only indices sampled without replacement by run seed, no labels; activation maxima",
              "selection": "Validation macro F1, earliest tied epoch; common budget, no early stopping, argmax at inference",
              "bundle_sha256": sha256(args.encoded / "bundle.json"),
              "source_sha256": {p.name: sha256(p) for p in source_paths},
              "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "deterministic_algorithms": True, "tf32": False,
              "limitations": ["Initial fixed-budget experiment, not an optimized model or proof of novelty.",
                              "Only 3 public test subjects, 2 validation subjects, and 1 personal webcam holdout.",
                              "RT-BENE source geometry/labels differ from personal eye-corner ROI collection.",
                              "Test/local holdout are reporting-only. Further tuning requires a new untouched evaluation set.",
                              "INT8 reference is specified integer inference, not measured Spartan-6 implementation."]}
    if encoders is not METHODS:
        config["experiment_scope"] = "Component ablation on declared splits; previously observed splits are replication/development, not a fresh unseen test"
    if getattr(args, "public_only", False):
        config.update(public_only=True, splits=list(active_splits),
                      protocol_sha256=args.protocol_sha256,
                      experiment_scope="Frozen public subject protocol; all declared seeds; no personal holdout",
                      limitations=["Only the declared public subject splits support new results.",
                                   "Precropped RGB and IR sources differ from live landmark crops.",
                                   "No novelty, superiority, medical or energy-efficiency claim follows from fitting.",
                                   "NumPy timings are not FPGA measurements."])
    if args.output.exists():
        if not args.resume or not (args.output / "config.json").is_file():
            raise FileExistsError(f"Refusing to overwrite run: {args.output}")
        if json.loads((args.output / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration/code/environment mismatch; use a new output")
    else:
        if args.stage == "evaluate":
            raise ValueError("Evaluation requires an existing frozen fit")
        args.output.mkdir(parents=True)
        write_json(args.output / "config.json", config)
    print(f"Device: {device}; GPU: {config['gpu']}; independent seeds: {args.seeds}", flush=True)
    if args.stage in ("fit", "all"):
        for method in args.methods:
            pixels = {split: load_pixels(args.encoded, method, split, len(partitions[split])) for split in ("train", "val")}
            for seed in args.seeds:
                fit_one(args.output / method / f"seed_{seed}", method, seed, pixels, partitions, args, device, encoders[method])
            del pixels
        selection = {f"{method}/seed_{seed}": json.loads((args.output / method / f"seed_{seed}" / "fit.json").read_text(encoding="utf-8"))
                     for method in args.methods for seed in args.seeds}
        # Check same initialization for each matching seed before permitting test evaluation.
        for seed in args.seeds:
            hashes = {selection[f"{method}/seed_{seed}"]["initial_state_sha256"] for method in args.methods}
            if len(hashes) != 1:
                raise ValueError("Methods did not start from matched independent initializations")
        write_json(args.output / "selection_frozen.json", selection)
    if args.stage in ("evaluate", "all"):
        frozen = json.loads((args.output / "selection_frozen.json").read_text(encoding="utf-8"))
        evaluations = {}
        for method in args.methods:
            pixels = {split: load_pixels(args.encoded, method, split, len(partitions[split])) for split in evaluation_splits}
            for seed in args.seeds:
                directory = args.output / method / f"seed_{seed}"
                record = frozen[f"{method}/seed_{seed}"]
                if sha256(directory / "float32.pt") != record["float32_sha256"] or sha256(directory / "int8_parameters.npz") != record["int8_sha256"]:
                    raise ValueError(f"Frozen parameters changed: {directory}")
                evaluations[f"{method}/seed_{seed}"] = evaluate_one(directory, method, pixels, partitions, args, device)
            del pixels
        summary = {"config": config, "runs": evaluations, "aggregate": {}}
        for method in args.methods:
            summary["aggregate"][method] = {}
            for split in evaluation_splits:
                summary["aggregate"][method][split] = {}
                for precision in ("float32", "int8"):
                    values = [evaluations[f"{method}/seed_{seed}"][split][precision] for seed in args.seeds]
                    summary["aggregate"][method][split][precision] = {
                        metric: {"mean": float(np.mean([r[metric] for r in values])),
                                 "std_population_across_seeds": float(np.std([r[metric] for r in values]))}
                        for metric in ("accuracy", "macro_f1", "balanced_accuracy")}
        write_json(args.output / "results.json", summary)
        with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "split", "precision", "accuracy_mean", "macro_f1_mean", "macro_f1_seed_std", "balanced_accuracy_mean"])
            for method, splits in summary["aggregate"].items():
                for split, precisions in splits.items():
                    for precision, values in precisions.items():
                        writer.writerow([method, split, precision, values["accuracy"]["mean"], values["macro_f1"]["mean"],
                                         values["macro_f1"]["std_population_across_seeds"], values["balanced_accuracy"]["mean"]])
        print(f"Complete: {args.output / 'results.json'}", flush=True)


def train_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS / "data")
    parser.add_argument("--encoded", type=Path, default=Path(ASSETS / "data/encoded/v1"))
    parser.add_argument("--output", type=Path, default=Path(ASSETS / "runs/phase4_v1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=["aeee2", "grayscale2"],
                        help="Fit/evaluate any path independently, using its own output directory")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage", choices=("fit", "evaluate", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Reuse completed, configuration-matching fits; never overwrite partial fits")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, OverflowError) as exc:
        parser.exit(1, f"Research run failed: {exc}\n")


# Train Personal

LABELS = {"OPEN": 0, "CLOSED": 1}
TARGETS = {"train": 32, "val": 16, "test": 16}


def reviewed_rows(root):
    """Require complete, operator-reviewed sessions with pair/session separation."""
    root = Path(root).resolve()
    collection = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    if not collection.get("complete"):
        raise ValueError("Guided collection is incomplete; training has not started")
    rows = validate_dataset(root)
    index = {row["sample_id"]: row for row in rows}
    accepted_ids, sessions, image_splits = set(), {}, {}
    result = {split: [] for split in TARGETS}
    for selected in collection["accepted"]:
        sample_id, split, label = selected["sample_id"], selected["split"], selected["label"]
        if sample_id in accepted_ids or split not in TARGETS or label not in LABELS:
            raise ValueError("Duplicate sample ID or invalid split/label in reviewed collection")
        accepted_ids.add(sample_id)
        if selected.get("reviewed_by_operator") is not True:
            raise ValueError("An unreviewed burst cannot be used for training/evaluation")
        row = index[sample_id]
        if row["subject_id"] != collection["subject_id"]:
            raise ValueError("Personal collection contains an unexpected subject")
        if row["session_id"] != collection["sessions"][split]:
            raise ValueError("A sample was moved across its original acquisition split")
        if row["left_label"] != label or row["right_label"] != label:
            raise ValueError("Reviewed labels differ from the original operator labels")
        session_key = (row["subject_id"], row["session_id"])
        if sessions.setdefault(session_key, split) != split:
            raise ValueError("One acquisition session appears in multiple splits")
        for side in ("left", "right"):
            digest = sha256(root / row[f"{side}_path"])
            if image_splits.setdefault(digest, split) != split:
                raise ValueError("An identical eye image appears in different splits")
        result[split].append(row)
    for split, minimum in TARGETS.items():
        counts = Counter(row["left_label"] for row in result[split])
        if any(counts[label] < minimum for label in LABELS):
            raise ValueError(f"Insufficient reviewed {split} pairs: {dict(counts)}")
    return collection, result


def encoded_batch(root, rows, method):
    arrays, labels = [], []
    for row in rows:
        for side in ("left", "right"):
            roi = read_roi(Path(root) / row[f"{side}_path"])
            roi.flags.writeable = False
            codes = METHODS[method].encode(roi)
            arrays.append(codes if method == "grayscale8" else codes * np.uint8(85))
            labels.append(LABELS[row[f"{side}_label"]])
    return np.stack(arrays)[:, None], np.asarray(labels, np.int64)


def logits(model, pixels):
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(pixels).float() / 255).numpy()


def fit_method(method, seed, root, rows, output, base_run, epochs, learning_rate, batch_size):
    # Loading the baseline checks encoder bundle, source, and both checkpoint hashes.
    baseline = EyePredictor(method, seed, base_run, compare_float=True)
    model = baseline.float_model
    train_x, train_y = encoded_batch(root, rows["train"], method)
    val_x, val_y = encoded_batch(root, rows["val"], method)
    train_tensor = torch.from_numpy(train_x).float() / 255
    labels_tensor = torch.from_numpy(train_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    directory = output / method / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    best_key = (-1.0, float("-inf"))
    best_epoch, history = None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        losses = []
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_tensor[indices]), labels_tensor[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = logits(model, val_x)
        scores = metrics(val_y, val_logits.argmax(axis=1))
        val_loss = float(criterion(torch.from_numpy(val_logits), torch.from_numpy(val_y)))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "validation_macro_f1": scores["macro_f1"], "validation_loss": val_loss}
        history.append(record)
        key = (scores["macro_f1"], -val_loss)
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save({"state_dict": model.state_dict(), "method": method, "seed": seed,
                        "encoder_version": METHODS[method].SPEC["version"], "epoch": epoch,
                        "architecture": ARCHITECTURE}, directory / "float32.pt")
        print(f"FIT {method} epoch={epoch}/{epochs} val_F1={scores['macro_f1']:.4f} val_loss={val_loss:.4f}", flush=True)
    selected = torch.load(directory / "float32.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    parameters, quantization = integer_model.make_parameters(model, train_tensor, batch_size)
    integer_model.save_parameters(directory, parameters, quantization)
    val_fp = logits(model, val_x)
    val_int = integer_model.infer(val_x, parameters)
    evaluation = {"val": {"float32": metrics(val_y, val_fp.argmax(axis=1)),
                           "int8": metrics(val_y, val_int.argmax(axis=1)),
                           "float_int8_prediction_agreement": float(np.mean(val_fp.argmax(axis=1) == val_int.argmax(axis=1)))}}
    fit = {"method": method, "seed": seed, "selected_epoch": best_epoch,
           "selected_validation_macro_f1": best_key[0], "history": history,
           "selection": "Maximum new-session validation macro F1; then minimum validation loss; then earliest epoch",
           "train_class_counts_OPEN_CLOSED": np.bincount(train_y, minlength=2).tolist(),
           "initial_checkpoint": str(baseline.directory / "float32.pt"),
           "initial_checkpoint_sha256": sha256(baseline.directory / "float32.pt"),
           "float32_sha256": sha256(directory / "float32.pt"),
           "int8_sha256": sha256(directory / "int8_parameters.npz"),
           "fit_and_calibration_seconds": time.perf_counter() - started,
           "scope": "personal adaptation; operator-labelled acquisition blocks; no unseen-subject claim"}
    write_json(directory / "fit.json", fit)
    write_json(directory / "evaluation.json", evaluation)
    return directory, evaluation


def personal_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, default=ASSETS / "runs/phase4_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1:
        parser.error("Use positive epochs/batch size and 0 < learning-rate < 1")
    root, output = args.data_dir.resolve(), args.output.resolve()
    collection, rows = reviewed_rows(root)
    original_manifest = ASSETS / "data/manifest.csv"
    original_hash = sha256(original_manifest)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.base_run / "config.json").read_text(encoding="utf-8"))
    config.update(purpose="personal adaptation on new operator-reviewed acquisition sessions",
                  epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                  seeds=[args.seed], data_directory=str(root),
                  personal_collection_sha256=sha256(root / "collection.json"),
                  personal_manifest_sha256=sha256(root / "manifest.csv"),
                  original_manifest_sha256=original_hash,
                  split_sessions=collection["sessions"], optimizer="Adam",
                  class_weighting="none; reviewed classes collected equally",
                  initial_weights="each method's own frozen phase4_v1 checkpoint",
                  calibration="all new TRAIN eyes only; no validation/test calibration",
                  selection="new validation macro F1, then minimum validation loss, then earliest epoch",
                  limitations=["Same user, nearby acquisition sessions, correlated pairs/frames",
                               "Personal adaptation; not subject-independent research evaluation",
                               "No changes to old public splits, local holdout or reported phase4_v1 results",
                               "Operator-reviewed labels; no automated ground-truth labels"])
    config["personal_source_sha256"] = {name: sha256(CODE_ROOT / name)
                                        for name in ("capture.py", "research.py")}
    write_json(output / "config.json", config)
    evaluations = {}
    for method in METHODS:
        directory, evaluation = fit_method(method, args.seed, root, rows, output,
                                           args.base_run, args.epochs, args.learning_rate, args.batch_size)
        evaluations[method] = evaluation
    frozen = {method: {name: sha256(output / method / f"seed_{args.seed}" / name)
                       for name in ("float32.pt", "int8_parameters.npz", "fit.json")}
              for method in METHODS}
    write_json(output / "selection_frozen.json", frozen)
    # First test inference happens only after ALL methods and integer calibration are frozen.
    for method in METHODS:
        directory = output / method / f"seed_{args.seed}"
        after = EyePredictor(method, args.seed, output, compare_float=True)
        before = EyePredictor(method, args.seed, args.base_run)
        test_x, test_y = encoded_batch(root, rows["test"], method)
        before_int = integer_model.infer(test_x, before.params)
        after_int = integer_model.infer(test_x, after.params)
        after_fp = logits(after.float_model, test_x)
        evaluations[method]["test"] = {
            "count": len(test_y), "before_int8": metrics(test_y, before_int.argmax(axis=1)),
            "int8": metrics(test_y, after_int.argmax(axis=1)),
            "float32": metrics(test_y, after_fp.argmax(axis=1)),
            "float_int8_prediction_agreement": float(np.mean(after_fp.argmax(axis=1) == after_int.argmax(axis=1))),
            "scope": "one same-user reserved acquisition session, correlated eye pairs"}
        write_json(directory / "evaluation.json", evaluations[method])
        print(f"TEST {method}: before_F1={evaluations[method]['test']['before_int8']['macro_f1']:.4f} "
              f"after_F1={evaluations[method]['test']['int8']['macro_f1']:.4f}", flush=True)
    if sha256(original_manifest) != original_hash:
        raise RuntimeError("The original research manifest changed during this run")
    write_json(output / "results.json", {"config": config, "runs": evaluations,
                                        "original_manifest_unchanged": True})
    print(f"PERSONAL TRAINING COMPLETE {output}", flush=True)
    return 0


# Aeee Ablation

class AEEEVariant:
    def __init__(self, removed):
        self.removed = removed
        self.SPEC = dict(aeee2.SPEC, version="aeee2-ablation-v1-no-"+removed,
                         removed_component=removed,
                         status="Component-removal control; no prior novelty or performance assumption")
        if removed == "adaptive_threshold":
            self.SPEC["threshold"] = "T=64, frozen original floor; not tuned on validation/test"
        elif removed == "central_band":
            self.SPEC["central_band"] = "W(y)=1 for all rows"
        elif removed == "continuity":
            self.SPEC["continuity"] = "No neighbor-support requirement"
        elif removed == "orientation":
            self.SPEC["eligible"] = "S>=T and abs(Gy)>0; no gradient-orientation filter"
        elif removed == "polarity":
            self.SPEC["symbols"] = {"0": "not retained", "1": "retained, S<2*T",
                                     "2": "retained, S>=2*T regardless of sign", "3": "unused"}
        else:
            raise ValueError("Unknown AEEE component: "+removed)

    def encode(self, roi):
        validate_roi(roi)
        gx, gy = sobel_components(roi)
        vertical = np.abs(gy)
        score = vertical.copy()
        if self.removed != "central_band":
            score[3:13] *= 2
        threshold = 64 if self.removed == "adaptive_threshold" else max(64, (int(score.sum(dtype=np.int64))+511)//512)
        eligible = (score >= threshold) & (vertical > 0)
        if self.removed != "orientation":
            eligible &= 2*vertical >= np.abs(gx)
        retained = eligible if self.removed == "continuity" else eligible & aeee2.horizontal_support(eligible)
        strong = retained & (score >= 2*threshold)
        codes = np.zeros((16, 32), np.uint8)
        codes[retained] = 1
        codes[strong] = 2
        if self.removed != "polarity":
            codes[strong & (gy < 0)] = 3
        return codes


COMPONENTS = ("central_band", "adaptive_threshold", "orientation", "continuity", "polarity")
METHODS_WITH_ABLATIONS = dict(METHODS, **{
    "aeee2_no_"+component: AEEEVariant(component) for component in COMPONENTS
})


# Ablation Research

def ablation_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "fit", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=ASSETS / "data")
    parser.add_argument("--encoded", type=Path, default=Path(ASSETS / "data/encoded/ablation_v1"))
    parser.add_argument("--output", type=Path, default=Path(ASSETS / "runs/ablation_v1"))
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS_WITH_ABLATIONS), default=list(METHODS_WITH_ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "encode":
            root = Path(__file__).resolve().parent
            build(args.data_dir, args.encoded, METHODS_WITH_ABLATIONS,
                  extra_sources=(root/"research.py",))
        else:
            run(args, METHODS_WITH_ABLATIONS)
    except (OSError, ValueError, KeyError, RuntimeError, OverflowError) as exc:
        parser.exit(1, f"Ablation stage failed: {exc}\n")
