"""Checked FPGA demo startup. --check never programs a device or opens a camera."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "assets/models/demo_deployment.json"


def checked_file(record, base):
    path = (base / record["path"]).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("missing or changed deployment file: " + str(path))
    return path


def preflight(catalog_path=CATALOG, model="auto", transport="auto", ports=None, probe_devices=True):
    """Inspect artifacts and read-only hardware identity; never program or open a camera."""
    result = dict(status="NOT_READY", camera="NOT_TESTED", device_firmware="NOT_VERIFIED",
                  programming_performed=False, candidates=[], selected=None)
    missing = [name for name in ("numpy", "cv2", "mediapipe", "serial") if importlib.util.find_spec(name) is None]
    result["runtime"] = "NOT_AVAILABLE" if missing else "READY"
    result["missing_modules"] = missing
    if not catalog_path.is_file():
        result["reason"] = "Deployment catalog unavailable; verified firmware and rollback must be registered first"
        return result
    if ports is None:
        if "serial" in missing:
            ports = []
        else:
            from serial.tools.list_ports import comports
            ports = [p.device for p in comports() if p.vid == 0x10c4 and p.pid == 0xea60]
    result["uart_devices"] = ports
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("schema_version") != 1:
            raise ValueError("unsupported deployment catalog")
        prerequisites = True
        try:
            landmarker = checked_file(catalog["landmarker"], catalog_path.parent)
            result["landmarker"] = dict(status="READY", sha256=catalog["landmarker"]["sha256"], path=str(landmarker))
        except (KeyError, OSError, ValueError) as exc:
            prerequisites = False
            result["landmarker"] = dict(status="NOT_READY", reason=str(exc))
        import tempfile
        try:
            directory = ROOT / "assets/runs"
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory) as stream:
                stream.write(b"preflight")
            result["report_directory"] = "READY"
        except OSError as exc:
            prerequisites = False
            result["report_directory"] = "NOT_READY"
        from importlib.metadata import version, PackageNotFoundError
        result["python_version"] = sys.version.split()[0]
        result["package_versions"] = {}
        for name in ("numpy", "opencv-python", "mediapipe", "pyserial"):
            try:
                result["package_versions"][name] = version(name)
            except PackageNotFoundError:
                result["package_versions"][name] = "NOT_AVAILABLE"
        for entry in catalog["images"]:
            candidate = dict(model=entry["model"], transport=entry["transport"], status="NOT_VERIFIED")
            result["candidates"].append(candidate)
            try:
                if entry["model"] not in ("tiny", "pretrained") or entry["transport"] not in ("jtag", "uart"):
                    raise ValueError("unsupported deployment contract")
                if entry.get("usercode") != (0x4a545931 if entry["model"] == "tiny" else 0x4a544e31):
                    raise ValueError("firmware identity differs from the selected protocol")
                for field in ("bit", "svf", "rollback_svf", "model_file", "physical_evidence"):
                    checked_file(entry[field], catalog_path.parent)
                if not entry.get("physical_verified"):
                    candidate["reason"] = "Physical inference validation pending"
                    continue
                evidence = json.loads(checked_file(entry["physical_evidence"], catalog_path.parent).read_text(encoding="utf-8"))
                if (evidence.get("bit_sha256") != entry["bit"]["sha256"] or
                        not all(evidence.get("gates", {}).get(key) is True for key in
                                ("hardware_identity", "physical_USER1", "resource_fit", "timing", "correctness"))):
                    raise ValueError("physical evidence does not accept this firmware digest")
                if entry["model"] == "pretrained":
                    if entry["model_file"]["sha256"] != "4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593":
                        raise ValueError("pretrained model digest differs from the accepted contract")
                else:
                    from live_pipeline import ModelSlot
                    slot = ModelSlot(checked_file(entry["manifest"], catalog_path.parent))
                    if not slot.ready or slot.method != "grayscale2" or slot.checkpoint_sha256 != entry["model_file"]["sha256"]:
                        raise ValueError("compact checkpoint, manifest and encoder mismatch")
                    if evidence.get("model_sha256") != slot.checkpoint_sha256:
                        raise ValueError("physical evidence was measured with a different compact model")
                from jtag_host import XSDB
                if not XSDB.is_file():
                    candidate.update(status="NOT_AVAILABLE", reason="XSDB volatile loader unavailable")
                    continue
                if missing or len(ports) != 1 or not prerequisites:
                    candidate.update(status="NOT_AVAILABLE", reason="runtime missing or CP2102 selection is not unique")
                    continue
                candidate["status"] = "READY"
                if model not in ("auto", entry["model"]) or transport not in ("auto", entry["transport"]):
                    continue
                rank = (entry["model"] == "tiny", entry["transport"] == "jtag")
                if result["selected"] is None or rank > selected_rank:
                    selected_rank = rank
                    result["selected"] = dict(entry, port=ports[0])
            except (KeyError, OSError, ValueError) as exc:
                candidate.update(status="NOT_READY", reason=str(exc))
        if result["selected"] is not None:
            result["status"] = "READY"
            if probe_devices:
                import os
                import subprocess
                from jtag_host import XSDB
                try:
                    inventory = subprocess.run([os.environ["COMSPEC"], "/d", "/c", str(XSDB), str(ROOT/"fpga/jtag_stream.tcl"), "discover"],
                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if inventory.returncode or "DISCOVERY_PASS" not in inventory.stdout:
                        raise RuntimeError(inventory.stdout[-1000:])
                    result["jtag_device"] = dict(status="READY", idcode="24001093", part="xc6slx9")
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    result.update(status="NOT_AVAILABLE", reason="FPGA discovery failed: "+str(exc))
                    result["selected"] = None
                if result["selected"] is not None:
                    from fpga_host import FPGAClient
                    client = None
                    try:
                        client = FPGAClient(ports[0], baud=1000000)
                        result["current_firmware"] = client.ping()
                        result["device_firmware"] = "READY"
                    except (OSError, RuntimeError, TimeoutError) as exc:
                        result["device_firmware"] = "NOT_VERIFIED"
                        result["firmware_reason"] = str(exc)
                    finally:
                        if client is not None:
                            client.close()
        else:
            result["reason"] = "No verified compatible deployment is available for this model/transport"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)
    return result


class DualClient:
    """One thread owns UART control and USER1 inference; close restores rollback."""
    def __init__(self, uart, session, inference):
        self.uart, self.session, self.inference = uart, session, inference

    def __getattr__(self, name):
        return getattr(self.uart, name)

    @property
    def supports_capture_ns(self):
        return getattr(self.inference, "supports_capture_ns", False)

    def load_model(self, model):
        import numpy as np
        status = self.uart.load_model(model)
        # Startup smoke inference precedes camera/landmarker construction. CPU
        # arithmetic verifies returned hardware logits and is never displayed.
        if model.method == "omz_bgr32":
            neutral = np.full((32,32,3), 127, np.uint8)
            result = self.inference.infer_pretrained(neutral, neutral, 0)
            expected = model.predict(neutral, fixed=True)
            method_id = 4
            if result.left_cycles != 367739 or result.right_cycles != 367739:
                raise RuntimeError("reference cycle count changed before camera startup")
        else:
            from fpga_host import model_image
            from integer_model import infer
            if model.method != "grayscale2" or self.uart.read_model() != model_image(model.params):
                raise RuntimeError("compact model readback mismatch before camera startup")
            neutral = np.full((16,32), 1, np.uint8)
            result = self.inference.infer(model.method, neutral, neutral, 0)
            expected = infer((neutral*np.uint8(85))[None,None],model.params)[0]
            method_id = 1
        if (result.frame_id != 0 or result.method_id != method_id
                or not np.array_equal(result.left_logits, expected)
                or not np.array_equal(result.right_logits, expected)):
            raise RuntimeError("FPGA startup arithmetic check failed before camera startup")
        return status

    def infer_pretrained(self, *args, **kwargs):
        return self.inference.infer_pretrained(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self.inference.infer(*args, **kwargs)

    def close(self):
        try:
            self.uart.close()
        finally:
            try:
                self.session.restore_baseline()
            finally:
                self.session.close()


def open_client(entry, base, log_path):
    """Explicit normal demo invocation authorizes volatile programming only."""
    from fpga_host import FPGAClient
    from jtag_host import JtagSession, JtagCNNClient
    compact = entry["model"] == "tiny"
    expected_code = 0x4a545931 if compact else 0x4a544e31
    if entry.get("usercode") != expected_code:
        raise ValueError("unsupported image contract")
    session = JtagSession(log_path, experimental_probe=True,
        program_probe_svf=checked_file(entry["svf"], base), expected_usercode=entry["usercode"],
        rollback_svf=checked_file(entry["rollback_svf"], base))
    uart = None
    try:
        uart = FPGAClient(entry["port"], baud=1000000)
        status = uart.ping()
        if (compact and status["method_id"] == 4) or (not compact and
                (not status.get("model_loaded") or status["method_id"] != 4 or status["fingerprint"] != 0x4daa1000)):
            raise RuntimeError("reference firmware/model identity mismatch before camera startup")
        if entry["transport"] == "jtag":
            session.set_frequency(12000000)  # Routed constraint and physical acceptance use 12 MHz.
        if compact:
            from tiny_jtag import JtagTinyClient
            inference = JtagTinyClient(session,int(entry["model_file"]["sha256"][:8],16))
        else:
            inference = JtagCNNClient(session)
        return DualClient(uart, session, inference if entry["transport"] == "jtag" else uart)
    except Exception:
        try:
            if uart is not None:
                uart.close()
        finally:
            try:
                session.restore_baseline()
            finally:
                session.close()
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["demo"]:
        argv.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--transport", choices=("auto", "jtag", "uart"), default="auto")
    parser.add_argument("--model", choices=("auto", "tiny", "pretrained"), default="auto")
    parser.add_argument("--deployment", type=Path, default=CATALOG)
    args = parser.parse_args(argv)
    report = preflight(args.deployment, args.model, args.transport)
    print(json.dumps(report, indent=2))
    if report["status"] != "READY" or args.check:
        return 0 if report["status"] == "READY" else 2
    # No GUI, landmarker, or camera initialization occurs above this gate.
    from uuid import uuid4
    from live_pipeline import live_cli
    output = ROOT / "assets/runs" / ("demo_" + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    entry = report["selected"]
    factory = lambda: open_client(entry, args.deployment.parent, output / "transport.log")
    factory.deployment = entry
    options = ["--network", entry["model"], "--backend", "fpga", "--precision", "fixed", "--pipeline", "overlap",
               "--baud", "1000000", "--port", entry["port"], "--report", str(output / "live.json")]
    if entry["model"] == "tiny":
        options += ["--manifest", str(checked_file(entry["manifest"], args.deployment.parent))]
    return live_cli(options,
                    forced_backend="fpga", client_factory=factory)


if __name__ == "__main__":
    raise SystemExit(main())
