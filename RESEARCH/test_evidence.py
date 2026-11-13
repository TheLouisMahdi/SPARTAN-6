"""Engineering fixtures only: synthetic metadata/arrays/video are never research findings."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PC"))
import cv2
import numpy as np
from encoders import METHODS, pack2, unpack2
from fpga_host import eye_payload, PacketReader, encode_packet
from data_tools import binary_label, sha256, build, load_partitions
from evidence import encoders, paired_bootstrap, PROTOCOL, sleepiness_baseline
from public_data import subject_splits, validate_rows, prepare, registered_videos
from ocular_metrics import summarize, window_summaries, observations, pts_milliseconds
from rtl_evidence import synthetic_parameters, layer_values
from integer_model import infer, requantize
from eye_state import BilateralBlinkDetector, StableEyeStates, BlinkEventLog, read_events


def samples(states, times=None, fractions=None):
    times = times or list(range(0, len(states) * 100, 100))
    return [dict(t_ms=t, left_state=s, right_state=s, left_open_fraction=None if fractions is None else fractions[i],
                 right_open_fraction=None if fractions is None else fractions[i]) for i, (t, s) in enumerate(zip(times, states))]


class BilateralBlinkTests(unittest.TestCase):
    def test_shared_event_log_can_be_evaluated(self):
        detector = BilateralBlinkDetector()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.csv"
            log = BlinkEventLog(path, "synthetic_joint")
            for i, state in enumerate(("OPEN", "OPEN", "CLOSED", "CLOSED", "OPEN", "OPEN")):
                event = detector.update(self.pair(state), i * 50)
                log.write({"both": event}, {"both": detector}, i * 50)
            log.close()
            self.assertEqual(read_events(path), {("synthetic_joint", "both"): [(150, 250)]})

    @staticmethod
    def pair(left="OPEN", right=None):
        return {"left": {"state": left}, "right": {"state": left if right is None else right}}

    def observe(self, detector, pairs, step=50):
        return [detector.update(pair, i * step) for i, pair in enumerate(pairs)]

    def test_one_shared_completed_cycle(self):
        detector = BilateralBlinkDetector()
        events = self.observe(detector, [self.pair(s) for s in
                              ("OPEN", "OPEN", "CLOSED", "CLOSED", "OPEN", "OPEN", "OPEN")])
        self.assertEqual(sum(events), 1)
        self.assertEqual(detector.count, 1)
        self.assertEqual(detector.last_blink_ms, 100)

    def test_winks_and_single_frame_errors_are_not_blinks(self):
        for changed in (self.pair("CLOSED", "OPEN"), self.pair("OPEN", "CLOSED")):
            detector = BilateralBlinkDetector()
            self.observe(detector, [self.pair(), self.pair(), changed, changed, self.pair(), self.pair()])
            self.assertEqual(detector.count, 0)
        detector = BilateralBlinkDetector()
        self.observe(detector, [self.pair(s) for s in ("OPEN", "OPEN", "CLOSED", "OPEN", "CLOSED", "OPEN", "OPEN")])
        self.assertEqual(detector.count, 0)

    def test_missing_mixed_or_gapped_observations_cancel_cycle(self):
        for interruption in ({}, self.pair("CLOSED", "OPEN")):
            detector = BilateralBlinkDetector()
            self.observe(detector, [self.pair(), self.pair(), self.pair("CLOSED"),
                                   self.pair("CLOSED"), interruption, self.pair(), self.pair()])
            self.assertEqual(detector.count, 0)
        detector = BilateralBlinkDetector()
        for t, s in ((0,"OPEN"),(50,"OPEN"),(100,"CLOSED"),(150,"CLOSED"),(500,"OPEN"),(550,"OPEN")):
            detector.update(self.pair(s), t)
        self.assertEqual(detector.count, 0)

    def test_unobserved_start_and_long_closure_are_not_blinks(self):
        detector = BilateralBlinkDetector()
        self.observe(detector, [self.pair(s) for s in ("CLOSED", "CLOSED", "OPEN", "OPEN")])
        self.assertEqual(detector.count, 0)
        detector = BilateralBlinkDetector()
        self.observe(detector, [self.pair()] * 2 + [self.pair("CLOSED")] * 16 + [self.pair()] * 2)
        self.assertEqual(detector.count, 0)

    def test_display_confirmation_does_not_change_raw_predictions(self):
        states = StableEyeStates()
        original = self.pair()
        self.assertEqual(states.update(original, 0)["left"]["state"], "UNAVAILABLE")
        self.assertEqual(states.update(original, 50)["left"]["state"], "OPEN")
        glitch = self.pair("CLOSED")
        self.assertEqual(states.update(glitch, 100)["left"]["state"], "OPEN")
        self.assertEqual(glitch["left"]["state"], "CLOSED")
        self.assertEqual(states.update(original, 150)["left"]["state"], "OPEN")
        states.update({}, 200)
        self.assertEqual(states.update(original, 250)["left"]["state"], "UNAVAILABLE")
        self.assertEqual(states.update(original, 300)["left"]["state"], "OPEN")


class EncodingTests(unittest.TestCase):
    def test_roundtrip_determinism_and_payload(self):
        roi = np.random.default_rng(12).integers(0, 256, (16, 32), np.uint8)
        for name, method in encoders(True).items():
            codes = method.encode(roi)
            np.testing.assert_array_equal(codes, method.encode(roi))
            if name != "grayscale8":
                payload = pack2(codes)
                self.assertEqual(len(payload), 128)
                np.testing.assert_array_equal(unpack2(payload), codes)
        self.assertEqual(pack2(np.tile(np.array([0, 1, 2, 3], np.uint8), (16, 8)))[0], 0x1b)
        self.assertEqual(len(eye_payload("aeee2", METHODS["aeee2"].encode(roi), METHODS["aeee2"].encode(roi))), 257)
        self.assertEqual(len(eye_payload("grayscale8", roi, roi)), 1025)

    def test_crc_and_sequence(self):
        good = encode_packet(0x92, 123, b"example")
        bad = bytearray(good)
        bad[-1] ^= 1
        reader = PacketReader()
        self.assertEqual(reader.feed(bytes(bad) + good[:5]), [])
        self.assertEqual(reader.feed(good[5:]), [(0x92, 123, b"example")])
        self.assertEqual(reader.bad_packets, 1)

    def test_integer_layers_saturation_and_relu(self):
        pixels = np.random.default_rng(1).integers(0, 256, (2, 1, 16, 32), np.uint8)
        for mode in ("mixed", "saturate", "relu_zero"):
            params = synthetic_parameters(mode=mode)
            values = layer_values(pixels, params)
            np.testing.assert_array_equal(values[-1], infer(pixels, params))
            self.assertEqual(values[1].shape, (2, 8, 2, 6))
            if mode == "relu_zero":
                self.assertFalse(np.any(values[0]))
        np.testing.assert_array_equal(requantize(np.array([-1, 0, 1, 600], np.int32), 1 << 23),
                                      np.array([-128, -128, -127, 127], np.int8))


class OcularTests(unittest.TestCase):
    def test_interblink_interval_does_not_bridge_missing_observations(self):
        rows = samples(["OPEN", "CLOSED", "OPEN", "UNAVAILABLE", "OPEN", "CLOSED", "OPEN"])
        eye = summarize(rows)["eyes"]["left"]
        self.assertEqual(eye["blink_count"], 2)
        self.assertIsNone(eye["mean_interblink_ms"])

    def test_variable_fps_event_and_no_binary_perclos(self):
        rows = samples(["OPEN", "CLOSED", "CLOSED", "OPEN", "OPEN"], [0, 40, 130, 230, 400])
        eye = summarize(rows)["eyes"]["left"]
        self.assertEqual(eye["blink_count"], 1)
        self.assertEqual(eye["median_blink_ms"], 190)
        self.assertAlmostEqual(eye["closed_fraction"], 190 / 400)
        self.assertIsNone(eye["perclos80_fraction"])
        self.assertIsNone(eye["blink_rate_per_min"])

    def test_aperture_needs_provenance(self):
        rows = samples(["OPEN", "CLOSED", "CLOSED", "OPEN"], fractions=[0.9, 0.1, 0.1, 0.9])
        self.assertIsNone(summarize(rows)["eyes"]["left"]["perclos80_fraction"])
        provenance = dict(subject_id="synthetic_fixture", calibration_method="fixture only", source_sha256="a" * 64)
        self.assertAlmostEqual(summarize(rows, aperture_provenance=provenance)["eyes"]["left"]["perclos80_fraction"], 2 / 3)

    def test_unavailable_gap_and_partial_events(self):
        for states, times in ((["OPEN", "CLOSED", "OPEN"], [0, 100, 500]),
                              (["OPEN", "CLOSED", "UNAVAILABLE", "OPEN"], [0, 100, 200, 300]),
                              (["CLOSED", "CLOSED", "OPEN"], [0, 100, 200]),
                              (["OPEN", "CLOSED", "CLOSED"], [0, 100, 200])):
            self.assertEqual(summarize(samples(states, times))["eyes"]["left"]["blink_count"], 0)
        eye = summarize(samples(["UNAVAILABLE"] * 4))["eyes"]["left"]
        self.assertIsNone(eye["closed_fraction"])
        self.assertIsNone(eye["blink_count"])
        self.assertEqual(eye["status"], "UNAVAILABLE")

    def test_false_short_blink_and_long_closure(self):
        eye = summarize(samples(["OPEN", "CLOSED", "OPEN"], [0, 10, 20]))["eyes"]["left"]
        self.assertEqual(eye["blink_count"], 0)
        rows = samples(["OPEN", "CLOSED", "CLOSED", "CLOSED", "OPEN"], [0, 100, 300, 500, 700])
        self.assertEqual(summarize(rows)["eyes"]["left"]["long_closure_count"], 1)

    def test_window_rates_and_reproducibility(self):
        rows = samples(["OPEN"] * 601)
        result = window_summaries(rows)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0]["partial_window"])
        self.assertEqual(result[0]["eyes"]["left"]["blink_rate_per_min"], 0)
        self.assertEqual(result, window_summaries(rows))
        self.assertTrue(window_summaries(rows[:30])[0]["partial_window"])

    def test_invalid_timestamps_and_fraction(self):
        for times in ([0, 0], [1, 0], [0, float("nan")], [-1, 10]):
            with self.assertRaises(ValueError):
                summarize(samples(["OPEN", "OPEN"], times))
        with self.assertRaises(ValueError):
            summarize(samples(["OPEN", "OPEN"], fractions=[1.1, 1.0]))

    def test_pts_strictness(self):
        for times in ((1, 1.04, 1.11), (1, 1, 1.1), (1, 0.9, 1.1)):
            process = type("Result", (), {"stdout": json.dumps({"frames": [dict(best_effort_timestamp_time=t) for t in times]})})()
            with patch("subprocess.run", return_value=process):
                if times[1] > times[0]:
                    self.assertEqual(pts_milliseconds("fixture"), [0, 40, 110])
                else:
                    with self.assertRaises(ValueError):
                        pts_milliseconds("fixture")


class PublicDataTests(unittest.TestCase):
    def test_labels_and_identity_split(self):
        self.assertEqual(binary_label("0.0"), "OPEN")
        self.assertEqual(binary_label("1.0"), "CLOSED")
        self.assertIsNone(binary_label("0.5"))
        subjects = [f"s{i}" for i in range(20)]
        self.assertEqual(subject_splits(subjects), subject_splits(subjects[::-1]))
        self.assertEqual(set(subject_splits(subjects).values()), {"train", "val", "test"})

    def test_leak_and_duplicates_rejected(self):
        row = dict(sample_id="a", subject_id="s1", session_id="v1", label="OPEN", label_scope="eye", split="train", canonical_sha256="a")
        for changes in (dict(sample_id="b", split="test", canonical_sha256="b"),
                        dict(sample_id="b", subject_id="s2"),
                        dict(sample_id="b", subject_id="s2", label="CLOSED")):
            with self.assertRaises(ValueError):
                validate_rows([row, dict(row, **changes)])

    def test_mrl_prepare_and_encoder_integration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            rng = np.random.default_rng(414)
            for person in range(5):
                for label in (0, 1):
                    cv2.imwrite(str(source / f"s{person:04d}_00001_0_0_{label}_0_0_01.png"), rng.integers(0, 256, (30, 60), np.uint8))
            registry = root / "registry.json"
            registry.write_text(json.dumps(dict(schema_version=1, datasets=[dict(dataset="MRL", version="SYNTHETIC_TEST_ONLY",
                source_url="https://mrl.cs.vsb.cz/eyedataset.html", license="synthetic test arrays, no faces", rights_note="test fixture",
                approved_local_use=True, root=str(source))])))
            prepared, encoded = root / "prepared", root / "encoded"
            info = prepare(registry, prepared)
            self.assertEqual(info["rows"], 10)
            partitions = load_partitions(prepared, ("train", "val", "test"))
            self.assertTrue(all(len(p) >= 2 for p in partitions.values()))
            build(prepared, encoded, splits=("train", "val", "test"))
            self.assertEqual(json.loads((encoded / "bundle.json").read_text())["shape"], [16, 32])

    def test_paired_subject_bootstrap(self):
        matrix = np.array([[[[10, 2], [1, 9]], [[8, 1], [2, 10]]]] * 5)
        result = paired_bootstrap(matrix, matrix, resamples=1000)
        self.assertEqual(result["delta_macro_f1"], 0)
        self.assertEqual(result["ci95_low"], 0)
        self.assertEqual(result, paired_bootstrap(matrix, matrix, resamples=1000))
        self.assertEqual(json.loads(PROTOCOL.read_text())["seeds"], [42, 43, 44, 45, 46])

    def test_train_only_sleepiness_scaling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "features.csv"
            rows = []
            for split, offset in (("train", 0), ("val", 100), ("test", 1000)):
                for label in (0, 5, 10):
                    row = dict(dataset="UTA-RLDD", subject_id=split + str(label), video_id=split + str(label),
                        source_sha256=hashlib.sha256((split + str(label)).encode()).hexdigest(),
                        label_scope="video", source_label=label, split=split)
                    for side in ("left", "right"):
                        row[side + "_closed_fraction"] = label / 10
                        row[side + "_blink_rate"] = label + offset
                    rows.append(row)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            result = sleepiness_baseline(path, root / "result.json")
            self.assertEqual(result["train_mean"], [0.5, 5.0, 0.5, 5.0])
            self.assertEqual(result["classes"], [0, 5, 10])

    @unittest.skipUnless(os.environ.get("EYE_RUN_TRAINING_FIXTURES") == "1", "Synthetic training requires explicit opt-in")
    def test_public_split_training_regression_synthetic_only(self):
        from research import run
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            (data / "research").mkdir(parents=True)
            (data / "images").mkdir()
            rng = np.random.default_rng(611)
            for split in ("train", "val", "test"):
                rows = []
                for label in ("OPEN", "CLOSED"):
                    name = split + label
                    path = data / "images" / (name + ".png")
                    cv2.imwrite(str(path), rng.integers(0, 256, (16, 32), np.uint8))
                    rows.append(dict(sample_id=name, subject_id=split, session_id=split, frame_group=name,
                                     split=split, label=label, roi_path=path.relative_to(data).as_posix()))
                with (data / "research" / (split + ".csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (data / "research/split_info.json").write_text('{"scope":"synthetic engineering fixture only"}')
            methods = {name: METHODS[name] for name in ("grayscale2", "aeee2")}
            encoded, output = root / "encoded", root / "run"
            build(data, encoded, methods, splits=("train", "val", "test"))
            args = SimpleNamespace(epochs=1, batch_size=2, learning_rate=0.001, seeds=[42], methods=list(methods),
                device="cpu", data_dir=data, encoded=encoded, output=output, stage="all", resume=False,
                splits=("train", "val", "test"))
            run(args, methods)
            self.assertTrue((output / "selection_frozen.json").exists())
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(set(results["runs"]["aeee2/seed_42"]), {"val", "test"})


class ReplayTests(unittest.TestCase):
    def test_missing_file_errors_without_fake_video(self):
        from app import replay_cli
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            with self.assertRaises(SystemExit):
                replay_cli([str(Path(temp) / "missing.avi"), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_blank_video_pts_headless_no_face(self):
        from app import replay_cli
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "blank.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                writer.write(np.zeros((240, 320, 3), np.uint8))
            writer.release()
            self.assertEqual(pts_milliseconds(video), [0, 100, 200, 300, 400, 500])
            output = Path(temp) / "out"
            self.assertEqual(replay_cli([str(video), "--output", str(output), "--timebase", "pts"]), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["frames_with_face"], 0)
            metric = json.loads((output / "ocular_metrics.json").read_text())
            self.assertIsNone(metric["summary"]["eyes"]["left"]["closed_fraction"])
            self.assertFalse(list(output.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
