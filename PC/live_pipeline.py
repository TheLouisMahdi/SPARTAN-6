"""Pretrained or explicit TinyEye inference with capture-timed bilateral events.

Default: pinned pretrained CNN. An explicitly selected empty TinyEye slot produces
UNAVAILABLE, never an OPEN/CLOSED guess or a MediaPipe expression-score fallback.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np

from blink_timing import BinocularTimer, MAXIMUM_GAP_MS
from encoders import METHODS
from encoders.common import validate_roi
from fpga_host import FPGAClient, model_image
from integer_model import infer

ROOT = Path(__file__).resolve().parents[1]
MODEL_SLOT = ROOT / "RESULTS/models/eye_model.json"
SIDES = ("left", "right")
LAYERS = ("conv1", "conv2", "fc1", "fc2")


def read_parameters(source):
    """Validate the existing INT8 UART contract before starting any device."""
    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    archive = np.load(BytesIO(payload), allow_pickle=False)
    if isinstance(archive, np.ndarray):
        raise ValueError("Checkpoint must be an NPZ archive, not a single NPY array")
    with archive:
        params = {name: archive[name] for name in archive.files}
    expected = {name + suffix for name in LAYERS for suffix in ("_weight", "_bias")}
    expected.update(name + "_multiplier" for name in LAYERS[:3])
    if set(params) != expected:
        raise ValueError("Checkpoint must contain the four layers' weights/biases and three multipliers")
    for name in LAYERS[:3]:
        multiplier = params[name + "_multiplier"]
        if multiplier.ndim != 0 or multiplier.dtype.kind not in "iu":
            raise ValueError("Q24 multipliers must be integer scalars")
    model_image(params)  # Shapes, dtypes, INT32 bounds and uint32 Q24 multipliers.
    for value in params.values():
        value.flags.writeable = False
    return params


class ModelSlot:
    """One explicitly activated checkpoint; architecture is a deployment contract."""

    def __init__(self, path=MODEL_SLOT):
        self.path = Path(path)
        self.record = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if (not isinstance(self.record, dict) or self.record.get("schema_version") != 1
                or self.record.get("architecture") != "tiny-eye-v1"
                or self.record.get("input") != [1, 16, 32]
                or self.record.get("classes") != ["OPEN", "CLOSED"]):
            raise ValueError("Model manifest does not match the current FPGA tensor contract")
        self.method = self.record.get("encoder")
        if not isinstance(self.method, str) or self.method not in METHODS:
            raise ValueError("Unknown model encoder")
        self.params = None
        self.checkpoint_sha256 = None
        if self.record.get("state") == "untrained":
            if self.record.get("checkpoint") is not None or self.record.get("sha256") is not None:
                raise ValueError("An untrained model slot must not reference a checkpoint")
            return
        if self.record.get("state") != "trained":
            raise ValueError("Model state must be untrained or trained")
        checkpoint = self.record.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("Trained model requires an explicit checkpoint path")
        if self.record.get("encoder_spec") != METHODS[self.method].SPEC:
            raise ValueError("Checkpoint encoder specification differs from the installed encoder")
        checkpoint = self.path.parent / checkpoint
        payload = checkpoint.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        if self.checkpoint_sha256 != self.record.get("sha256"):
            raise ValueError("Checkpoint hash differs from the activated model")
        self.params = read_parameters(payload)

    @property
    def ready(self):
        return self.params is not None


@dataclass(frozen=True)
class PairResult:
    frame_id: int
    capture_ns: int
    eyes: dict
    processing_ms: float
    hardware_cycles: tuple | None = None
    backends: dict | None = None


class EyePipeline:
    """Return both eye results for one original camera frame."""

    def __init__(self, model, workers=2, client=None, verify=False):
        if workers not in (1, 2):
            raise ValueError("Eye workers must be 1 or 2")
        if client is not None and not model.ready:
            raise ValueError("FPGA inference requires an explicitly trained model")
        self.model, self.client, self.verify = model, client, verify
        cpu_ready = model.ready and client is None
        self.batch_eyes = cpu_ready and model.method == "omz_bgr32" and workers == 2
        self.pool = ThreadPoolExecutor(max_workers=workers) if cpu_ready and not self.batch_eyes else None

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def _codes(self, roi):
        validate_roi(roi)
        return METHODS[self.model.method].encode(roi)

    def _infer_eye(self, roi):
        if self.model.method == "omz_bgr32":
            return self.model.predict(roi)
        codes = self._codes(roi)
        pixels = codes if self.model.method == "grayscale8" else codes * np.uint8(85)
        return infer(pixels[None, None], self.model.params)[0]

    def predict(self, frame_id, capture_ns, rois):
        started = time.perf_counter_ns()
        eyes = {side: {"state": "UNAVAILABLE", "logits": None, "margin": None} for side in SIDES}
        available = {side: rois[side] for side in SIDES if rois.get(side) is not None}
        scores = {}
        hardware_cycles = None
        if self.model.ready and self.client is not None:
            if len(available) == 2:
                if self.model.method == "omz_bgr32":
                    pair = [available[side] for side in SIDES]
                    if getattr(self.client, "supports_capture_ns", False):
                        hardware = self.client.infer_pretrained(*pair, frame_id, capture_ns=capture_ns)
                    else:
                        hardware = self.client.infer_pretrained(*pair, frame_id)
                else:
                    codes = [self._codes(available[side]) for side in SIDES]
                    if getattr(self.client,"supports_capture_ns",False):
                        hardware = self.client.infer(self.model.method, *codes, frame_id, capture_ns=capture_ns)
                    else:
                        hardware = self.client.infer(self.model.method, *codes, frame_id)
                hardware_cycles = (hardware.left_cycles, hardware.right_cycles, hardware.pair_cycles)
                scores = dict(zip(SIDES, (hardware.left_logits, hardware.right_logits)))
                if self.verify:
                    for side in SIDES:
                        reference = (self.model.predict(available[side], fixed=True) if self.model.method == "omz_bgr32"
                                     else self._infer_eye(available[side]))
                        if not np.array_equal(scores[side], reference):
                            raise RuntimeError("FPGA logits differ from the independent integer reference")
            else:
                self.client.invalidate(frame_id)
        elif self.model.ready and len(available) == 2:
            if self.batch_eyes:
                scores = dict(zip(SIDES, self.model.predict_pair(available["left"], available["right"])))
            else:
                jobs = {side: self.pool.submit(self._infer_eye, roi) for side, roi in available.items()}
                scores = {side: job.result() for side, job in jobs.items()}
        for side, logits in scores.items():
            convert = float if self.model.method == "omz_bgr32" and self.client is None else int
            logits = [convert(value) for value in logits]
            margin = logits[1] - logits[0]
            # Equal logits give no evidence for either state. Scores are not probabilities.
            state = "CLOSED" if margin > 0 else "OPEN" if margin < 0 else "UNAVAILABLE"
            eyes[side] = {"state": state, "logits": logits, "margin": margin}
        return PairResult(frame_id, capture_ns, eyes, (time.perf_counter_ns() - started) / 1e6, hardware_cycles)


def update_timing(timer, result, now_ns, max_age_ms=MAXIMUM_GAP_MS):
    """Freshness is checked at acceptance; unavailable inputs cancel partial cycles."""
    fresh = 0 <= now_ns - result.capture_ns <= max_age_ms * 1_000_000
    scores = [None if not fresh or result.eyes[side]["state"] == "UNAVAILABLE"
              else float(result.eyes[side]["state"] == "CLOSED") for side in SIDES]
    timing = timer.update(*scores, (result.capture_ns // 1_000_000) & 0xffffffff)
    return timing, scores


def synchronize_timing(timer, result, client=None):
    """Mirror capture-time events; hide expired live states after a late reply.

    Invalidation cancels continuity, not completed history. A fresh observation
    accepted before UART dispatch can already have committed an event on the
    FPGA, so a delayed acknowledgement must not be presented as a rollback.
    """
    timing, scores = update_timing(timer, result, time.monotonic_ns())
    if client is not None:
        hardware, _ = client.eye_timing(*scores, (result.capture_ns // 1_000_000) & 0xffffffff, result.frame_id)
        if hardware != timing:
            raise RuntimeError("FPGA binocular timing differs from the capture-timed reference")
        timing = hardware
    finished_ns = time.monotonic_ns()
    if finished_ns - result.capture_ns > MAXIMUM_GAP_MS * 1_000_000:
        timer.invalidate()
        if client is not None:
            client.invalidate(result.frame_id)
        scores = [None, None]
    return timing, scores, finished_ns


def model_cli(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the empty model slot or explicitly activate future INT8 weights.")
    parser.add_argument("--manifest", type=Path, default=MODEL_SLOT)
    parser.add_argument("--parameters", type=Path, help="Explicit trained int8_parameters.npz; does not train or measure accuracy")
    parser.add_argument("--encoder", choices=tuple(METHODS), help="Required with --parameters; must match its training preprocessing")
    args = parser.parse_args(argv)
    if bool(args.parameters) != bool(args.encoder):
        parser.error("--parameters and --encoder must be supplied together")
    try:
        if args.parameters:
            payload = args.parameters.read_bytes()
            read_parameters(payload)
            try:
                checkpoint_reference = os.path.relpath(args.parameters.resolve(), args.manifest.resolve().parent)
            except ValueError:  # Windows paths on different drives.
                checkpoint_reference = str(args.parameters.resolve())
            record = {"schema_version": 1, "state": "trained", "architecture": "tiny-eye-v1",
                      "input": [1, 16, 32], "classes": ["OPEN", "CLOSED"], "encoder": args.encoder,
                      "encoder_spec": METHODS[args.encoder].SPEC,
                      "checkpoint": checkpoint_reference, "sha256": hashlib.sha256(payload).hexdigest(),
                      "accuracy": "not established by activation"}
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.manifest.with_name(args.manifest.name + ".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            try:
                model = ModelSlot(temporary)
                temporary.replace(args.manifest)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            model = ModelSlot(args.manifest)
        print(json.dumps({"state": model.record["state"], "ready": model.ready,
                          "architecture": model.record["architecture"], "encoder": model.method,
                          "checkpoint_sha256": model.checkpoint_sha256}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"MODEL ERROR: {exc}\n")


def render_stream_preview(snapshot, now_ns, counts, first_capture_ns, mode):
    """Render an owned frame without opening a window, camera or landmarker."""
    import cv2
    item=snapshot.inference.prepared; display=item.frame.copy()
    expired=now_ns-item.capture_ns>MAXIMUM_GAP_MS*1000000
    for side,box in item.boxes.items():
        x0,y0,x1,y1=box; cv2.rectangle(display,(x0,y0),(x1,y1),(80,180,180),1)
    elapsed=max(.001,(now_ns-first_capture_ns)/1e9)
    kernel=(snapshot.inference.hardware.left_cycles+snapshot.inference.hardware.right_cycles)/50000 if snapshot.inference.hardware else None
    lines=['SPARTAN-6 | CPU-FPGA PIPELINE',f'Backend FPGA P2 | {mode.upper()} | frame {item.frame_id}',
        ' | '.join(s+': '+('UNAVAILABLE' if expired else snapshot.eyes[s]['state']) for s in SIDES),
        f'Prep {(item.prep_end_ns-item.prep_start_ns)/1e6:.2f} ms | Transport {(snapshot.inference.end_ns-snapshot.inference.start_ns)/1e6:.2f} ms',
        f'Age {(now_ns-item.capture_ns)/1e6:.1f} ms | blinks {snapshot.timing.count}',
        f'Capture {counts.get("capture",0)/elapsed:.1f} FPS | processed {counts.get("fpga_completed",0)/elapsed:.2f} FPS',
        f'Kernel pair {kernel} ms | prep drops {counts.get("dropped_preparation",0)} | RUNNING {snapshot.reason or ""}']
    for n,line in enumerate(lines): cv2.putText(display,line,(8,25+22*n),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,235,180),1,cv2.LINE_AA)
    return display


def run_streaming(args, model, client_factory=None):
    """The GUI remains on the caller. All UART activity belongs to eye-fpga."""
    import csv
    from dataclasses import asdict
    import subprocess
    from uuid import uuid4
    from queue import Empty, Queue
    from threading import Event, Thread
    import cv2
    from capture import capture_frames, extract_eye_roi, EYE_CORNERS, MODEL_PATH, mp, vision
    from stream_pipeline import StreamingPipeline, process_rss_bytes, offer_latest
    from collections import deque
    from pipeline_report import summarize
    from experiment import write_json
    from bounded_log import BoundedLog
    report=args.report
    if report == ROOT/'RESULTS/logs/live/latest.json':
        report=ROOT/'RESULTS/runs'/('pipeline_'+args.pipeline+'_'+uuid4().hex[:12])/'session.json'
    directory=report.parent
    if report.exists() or (directory/'trace.jsonl').exists() or (directory/'summary.json').exists() or (args.frames and args.frames.exists()):
        raise FileExistsError('Pipeline outputs must be new; existing reports are preserved')
    directory.mkdir(parents=True,exist_ok=True)
    if args.frames: args.frames.parent.mkdir(parents=True,exist_ok=True)
    replay_rows=[]
    pretrained = model.method == 'omz_bgr32'
    capture_capacity = 1 if pretrained else 2
    if args.replay_manifest:
        from public_data import read_cew_manifest
        replay_rows=sorted(read_cew_manifest(args.replay_manifest),key=lambda r:r['source_sha256'])[:args.replay_count*2]
        if len(replay_rows)<args.replay_count*2: raise ValueError('Insufficient registered replay inputs')
    replay_frames=[]; replay_preload_ms=0
    if replay_rows:
        from experiment import read_color_image
        from data_tools import sha256
        preload_start=time.perf_counter_ns()
        for i in range(args.replay_count):
            pair=[]
            for row in replay_rows[2*i:2*i+2]:
                if sha256(Path(row['source_path']))!=row['source_sha256']: raise ValueError('Replay source hash changed')
                image,_=read_color_image(row['source_path'])
                pair.append(cv2.resize(image,(32,32),interpolation=cv2.INTER_LINEAR) if pretrained else
                            cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,16),interpolation=cv2.INTER_AREA))
            fixture=np.concatenate(pair,axis=1);fixture.flags.writeable=False;replay_frames.append(fixture)
        replay_preload_ms=(time.perf_counter_ns()-preload_start)/1e6
    class Prepare:
        def __init__(self):
            self.landmarker=None
            if not replay_rows:
                options=vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),delegate=mp.tasks.BaseOptions.Delegate.CPU),
                    running_mode=vision.RunningMode.VIDEO,num_faces=1)
                self.landmarker=vision.FaceLandmarker.create_from_options(options)
        def __call__(self,frame,capture_ns):
            if replay_rows:
                rois={'left':frame[:,:32], 'right':frame[:,32:]}; boxes={}
            else:
                start=time.perf_counter_ns()
                found=self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),capture_ns//1000000)
                landmark_end=time.perf_counter_ns()
                rois,boxes={},{}
                if found.face_landmarks:
                    for side,corners in EYE_CORNERS.items():
                        gray,box=extract_eye_roi(frame,found.face_landmarks[0],corners)
                        if box is not None:
                            x0,y0,x1,y1=box; boxes[side]=box; rois[side]=frame[y0:y1,x0:x1].copy() if pretrained else gray
                pipe.record('preparation_components',capture_ns=capture_ns,landmark_start_ns=start,landmark_end_ns=landmark_end,roi_end_ns=time.perf_counter_ns())
            # Identical BGR interpolation is performed before handing work to UART.
            if pretrained:
                from pretrained_eye import input_pixels
                rois={s:np.ascontiguousarray(input_pixels(roi).transpose(1,2,0)) for s,roi in rois.items()}
            return rois,boxes
        def close(self):
            if self.landmarker is not None: self.landmarker.close()
    def source(stop):
        if replay_rows:
            pending,errors=Queue(capture_capacity),Queue(1);ended=Event()
            def produce():
                start=time.perf_counter_ns()
                try:
                    for i in range(args.replay_count):
                        captured=start+round(i*1e9/args.replay_fps)
                        if stop.wait(max(0,(captured-time.perf_counter_ns())/1e9)): break
                        pipe.record('capture',frame_id=i+1,capture_ns=captured,source_ids=[r['sample_id'] for r in replay_rows[2*i:2*i+2]])
                        old=offer_latest(pending,(i+1,captured,replay_frames[i]))
                        if old is not None:pipe.record('dropped_capture',frame_id=old[0],reason='ROI replay latest-frame')
                except Exception as exc: errors.put_nowait(exc)
                finally: ended.set()
            worker=Thread(target=produce,name='roi-source',daemon=True);worker.start()
            try:
                while not stop.is_set():
                    try:yield pending.get(timeout=.05)
                    except Empty:
                        if stop.is_set(): break
                        if ended.is_set():
                            if not errors.empty():raise errors.get_nowait()
                            break
            finally:
                stop.set();worker.join(3)
                if worker.is_alive():raise RuntimeError('ROI source did not close')
                while not pending.empty():pipe.record('dropped_capture',frame_id=pending.get_nowait()[0],reason='shutdown')
            return
        pending,errors=Queue(capture_capacity),Queue(1); stopped=Event()
        worker=Thread(target=capture_frames,args=(pending,errors,stop,stopped,args.camera,
            lambda fid,ns:pipe.record('capture',frame_id=fid,capture_ns=ns),
            lambda fid,reason:pipe.record('dropped_capture',frame_id=fid,reason=reason),time.perf_counter_ns),name='camera-capture',daemon=True)
        worker.start()
        try:
            while not stop.is_set():
                try: yield pending.get(timeout=.05)
                except Empty:
                    if stop.is_set(): break
                    if stopped.is_set(): raise RuntimeError(errors.get_nowait() if not errors.empty() else 'Camera stopped')
        finally:
            stop.set(); worker.join(3)
            if worker.is_alive(): raise RuntimeError('Camera release did not complete')
            while not pending.empty(): pipe.record('dropped_capture',frame_id=pending.get_nowait()[0],reason='shutdown')
    def emit(event): trace.write(event)
    startup_timeout = 75 if client_factory is not None else 5
    pipe=StreamingPipeline(source,Prepare,client_factory or (lambda:FPGAClient(args.port,baud=args.baud or 1000000)),model,
        mode=args.pipeline,verify=args.verify,emit=emit,startup_timeout=startup_timeout,prepared_capacity=1 if pretrained else 4)
    meta=dict(schema_version=1,report_kind='stream_pipeline_v1',mode=args.pipeline,source='ROI_REPLAY_ONLY' if replay_rows else 'LIVE_CAMERA_MEDIAPIPE',
        provenance='PHYSICAL_MEASUREMENT',model_sha256=model.checkpoint_sha256,output_order=['OPEN','CLOSED'],
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        bitstream_sha256_local_artifact=(getattr(client_factory,'deployment',{}).get('bit',{}).get('sha256') or
            (hashlib.sha256((ROOT/'FPGA/eye_top.bit').read_bytes()).hexdigest() if (ROOT/'FPGA/eye_top.bit').exists() else None)),
        bitstream_provenance='Deployment digest with startup arithmetic/readback gates when supplied; otherwise local artifact only',
        port=args.port,baud=args.baud or 1000000,camera=None if replay_rows else args.camera,verify=args.verify,
        warmup_seconds=args.warmup_seconds,start_ns=time.perf_counter_ns(),status='PASS',timing_source='CPU BinocularTimer; onboard diagnostic available through legacy mode',
        gui=not args.headless,frames_path=str(args.frames or directory/'frames.csv'),images_saved=False,
        capture_and_stage_clock='time.perf_counter_ns / Windows QPC, monotonic high resolution; legacy capture default unchanged',
        requested_seconds=args.seconds)
    meta.update(source_sha256={name:hashlib.sha256((ROOT/'PC'/name).read_bytes()).hexdigest()
        for name in ('live_pipeline.py','stream_pipeline.py','capture.py','fpga_host.py','pretrained_eye.py')},
        queue_capacities=dict(capture=capture_capacity,prepared=pipe.prepared.maxsize,inference=2,display=1),serial_owner='eye-fpga',
        preparation_owner='eye-prepare',result_owner='eye-result',process_id=__import__('os').getpid())
    if args.replay_manifest:
        meta.update(replay_manifest_sha256=hashlib.sha256(args.replay_manifest.read_bytes()).hexdigest(),
            replay_count=args.replay_count,replay_fps=args.replay_fps,replay_preload_ms=replay_preload_ms,fixture_preload_bytes=sum(frame.nbytes for frame in replay_frames),
            replay_timestamp_definition='synthetic fixed-rate capture epoch; trace at_ns records actual dispatch, including scheduler lateness',
            replay_selection='Source SHA256 order; first 2*count eye patches; no label/model-based selection')
    exit_code=0; first_capture=None; duration_completed=False; ui_polls=0; ui_last=None; ui_max_gap_ms=None
    memory_samples=deque(maxlen=64);next_memory=0;latest_snapshot=None
    window='SPARTAN-6 | CPU-FPGA PIPELINE'
    trace=BoundedLog(directory/'trace.jsonl',formatter=lambda event:json.dumps(event)+'\n')
    try:
        pipe.start()
        if not args.headless: cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        while not pipe.done.is_set():
            pipe.raise_error()
            observed=time.perf_counter_ns()
            if observed>=next_memory:
                memory_samples.append(dict(at_ns=observed,rss_bytes=process_rss_bytes()));next_memory=observed+10_000_000_000
            if first_capture is None and pipe.first_capture_ns is not None: first_capture=pipe.first_capture_ns
            if first_capture is not None and args.seconds and (time.perf_counter_ns()-first_capture)/1e9>=args.seconds:
                duration_completed=True;break
            if first_capture is None and time.perf_counter_ns()-meta['start_ns']>max(15,startup_timeout)*1e9: raise TimeoutError('No camera/source frame within startup deadline')
            try: snapshot=pipe.display.get(timeout=.03)
            except Empty: snapshot=None
            if snapshot is not None: latest_snapshot=snapshot
            if latest_snapshot is not None and not args.headless:
                display=render_stream_preview(latest_snapshot,time.perf_counter_ns(),pipe.counts,pipe.first_capture_ns or meta['start_ns'],args.pipeline)
                cv2.imshow(window,display)
            if not args.headless:
                ui_now=time.perf_counter_ns();ui_polls+=1
                if ui_last is not None: ui_max_gap_ms=max(ui_max_gap_ms or 0,(ui_now-ui_last)/1e6)
                ui_last=ui_now
                pipe.record('ui_pump',thread_id=__import__('threading').get_ident())
                if cv2.waitKey(1)&255 in (ord('q'),ord('Q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    if args.seconds: meta['status']='INTERRUPTED';exit_code=130
                    break
        pipe.raise_error()
    except KeyboardInterrupt:
        meta['status']='INTERRUPTED';exit_code=130
    except Exception as exc:
        meta.update(status='FAIL',error=str(exc)); exit_code=1
    finally:
        try: pipe.close(timeout=max(6,startup_timeout)); pipe.raise_error()
        except Exception as exc: meta.update(status='FAIL',shutdown_error=str(exc));exit_code=1
        meta.update(end_ns=time.perf_counter_ns(),identity=pipe.identity,integrity=pipe.integrity,clean_shutdown=not any(w.is_alive() for w in pipe.threads),
            completed_requested_duration=duration_completed,ui_poll_count=ui_polls,ui_max_poll_gap_ms=ui_max_gap_ms,
            ui_thread_id=__import__('threading').get_ident(),memory_samples=list(memory_samples))
        try: trace.close()
        except Exception as exc: meta.update(status='FAIL',trace_error=str(exc));exit_code=1
        meta['trace_writer_clean_shutdown']=not trace.worker.is_alive()
        if not args.headless: cv2.destroyAllWindows()
        write_json(report,meta)
        summarize(directory,meta,args.frames)
    print(json.dumps(dict(status=meta['status'],report=str(report),summary=str(directory/'summary.json'))),flush=True)
    return exit_code


def live_cli(argv=None, forced_backend=None, client_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Explicit legacy TinyEye manifest; cannot be combined with --network pretrained")
    parser.add_argument("--network", choices=("pretrained", "tiny"), help="Default: pretrained; explicit --manifest selects TinyEye")
    parser.add_argument("--precision", choices=("float", "fixed"), default="float", help="CPU pretrained arithmetic; FPGA always uses fixed")
    parser.add_argument("--baud", type=int, help="Default: 1000000 for pretrained firmware, 115200 for TinyEye")
    parser.add_argument("--backend", choices=("cpu", "fpga", "compare"), default=forced_backend or "cpu")
    parser.add_argument("--encoder", choices=("aeee2", "grayscale2"), help="Select an independent TinyEye research model")
    parser.add_argument("--output", type=Path, help="New experiment session directory")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2,
                        help="Pretrained CPU: 2 batches both eyes (default), 1 runs each eye separately. TinyEye CPU: worker count")
    parser.add_argument("--camera", type=int, default=1, help="Integrated Camera on this laptop; choose explicitly if devices change")
    parser.add_argument("--port", default="COM26", help="Opened only with --backend fpga")
    parser.add_argument("--verify", action="store_true", help="Also recompute FPGA logits on CPU; verification overhead is included")
    parser.add_argument("--seconds", type=float, default=0, help="0: run until Q/Esc")
    parser.add_argument("--status", action="store_true", help="Show readiness without opening camera or UART")
    parser.add_argument("--report", type=Path, default=ROOT / "RESULTS/logs/live/latest.json")
    parser.add_argument("--frames", type=Path, help="New prediction CSV for offline ocular analysis; never ground-truth labels")
    parser.add_argument("--pipeline", choices=("sequential", "overlap"), help="Explicit bounded pretrained FPGA stages; omitted preserves the diagnostic baseline")
    parser.add_argument("--headless", action="store_true", help="No OpenCV windows in an explicit pipeline run")
    parser.add_argument("--warmup-seconds", type=float, default=3, help="Exclude initial pipeline warmup from throughput statistics")
    parser.add_argument("--replay-manifest", type=Path, help="Registered CEW manifest for ROI_REPLAY_ONLY, never a camera benchmark")
    parser.add_argument("--replay-count", type=int, default=20)
    parser.add_argument("--replay-fps", type=float, default=4, help="Identical host-clock ROI replay pacing for both modes")
    args = parser.parse_args(argv)
    experiment = args.encoder is not None or args.backend == "compare" or args.output is not None
    if experiment and not args.encoder:
        parser.error("Experiment sessions require --encoder aeee2 or grayscale2")
    if experiment and args.network == "pretrained":
        parser.error("Research encoders require TinyEye")
    if experiment:
        args.network = "tiny"
        args.manifest = args.manifest or ROOT / "RESULTS/models" / args.encoder / "manifest.json"
        if args.output is not None and args.output.exists():
            parser.error("--output must be a new session directory")
        if args.backend != "cpu" and not args.port.strip():
            parser.error("A nonempty UART port is required")
    if args.camera < 0 or not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("Camera must be nonnegative; seconds must be finite and nonnegative")
    if forced_backend and args.backend != forced_backend:
        parser.error(f"This command requires --backend {forced_backend}")
    if args.verify and args.backend != "fpga":
        parser.error("--verify requires the FPGA backend")
    if args.manifest and args.network == "pretrained":
        parser.error("--manifest refers to TinyEye; omit it for the pretrained CNN")
    pretrained = args.network == "pretrained" or (args.network is None and args.manifest is None)
    if args.baud is not None and args.baud <= 0:
        parser.error("Baud rate must be positive")
    try:
        if pretrained:
            from pretrained_eye import PretrainedEye
            model = PretrainedEye(precision=args.precision)
        else:
            model = ModelSlot(args.manifest or MODEL_SLOT)
        if experiment and (not model.ready or model.method != args.encoder):
            raise ValueError("Trained checkpoint and encoder must match the selected experiment")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{'MODEL_NOT_READY' if experiment else 'MODEL ERROR'}: {exc}\n")
    print(f"MODEL: {model.record['architecture']}; {'ready' if model.ready else 'NOT TRAINED; eye states unavailable'}", flush=True)
    if args.status:
        cpu_mode = "; paired batch" if pretrained and args.backend == "cpu" and args.workers == 2 else ""
        print(f"BACKEND: {args.backend}{cpu_mode}; camera/UART not opened")
        return 0
    if args.pipeline:
        if args.backend != 'fpga' or experiment or not model.ready:
            parser.error('--pipeline requires a trained FPGA model without experiment mode')
        if args.warmup_seconds<0 or not np.isfinite(args.warmup_seconds) or args.replay_count<1 or not np.isfinite(args.replay_fps) or args.replay_fps<=0:
            parser.error('Invalid pipeline warmup or replay configuration')
        return run_streaming(args,model,client_factory)
    if args.headless or args.replay_manifest:
        parser.error('--headless/--replay-manifest require an explicit --pipeline')
    if args.backend == "fpga" and not model.ready:
        parser.exit(1, "Train and explicitly activate a model before FPGA inference. No devices opened.\n")

    # Heavy device imports and initialization happen only after the readiness checks.
    import cv2
    from queue import Empty, Queue
    from threading import Event, Thread
    from capture import EYE_CORNERS, MODEL_PATH, capture_frames, extract_eye_roi, mp, vision, text

    pending, errors = Queue(maxsize=1), Queue(maxsize=1)
    stop, stopped = Event(), Event()
    camera_worker = landmarker = client = pipeline = session = None
    timer = BinocularTimer()
    frame_stream = writer = None
    frame_origin_ms = None
    stats = {"schema_version": 1, "backend": args.backend, "workers": (0 if args.backend == "fpga" else 1) if experiment else args.workers,
             "model_state": model.record["state"], "checkpoint_sha256": model.checkpoint_sha256,
             "encoder": model.method, "camera": args.camera, "verify_cnn_on_cpu": args.verify,
             "architecture": model.record["architecture"],
             "cpu_precision": (args.precision if args.backend == "cpu" else "fixed" if args.verify else None) if pretrained else "int8",
             "baud": args.baud or (1000000 if pretrained else 115200),
             "frames": 0, "valid_eye_pairs": 0, "skipped_capture_frames": 0,
             "hardware_cnn_pairs": 0, "last_hardware_cycles": None,
             "hardware_clock_hz_from_project": 50000000,
             "pair_processing_sum_ms": 0.0, "pair_processing_max_ms": 0.0,
             "after_read_sum_ms": 0.0, "after_read_max_ms": 0.0,
             "images_saved": False, "training_performed": False, "calibration_performed": False,
             "accuracy": "not measured; prediction CSV is not ground truth",
             "timing": "capture timestamps; freshness at acceptance; expired display states are hidden",
             "aperture": "unavailable; binary logits do not measure eye-opening fraction"}
    previous_frame_id = 0
    window = "Eye pipeline | Q / Esc to quit"
    exit_code = 0
    try:
        if experiment:
            from experiment import Session
            session = Session(args.output, model, args.backend, args.camera, args.port, stats["baud"])
        if args.frames:
            args.frames.parent.mkdir(parents=True, exist_ok=True)
            frame_stream = args.frames.open("x", encoding="utf-8", newline="")
            writer = csv.DictWriter(frame_stream, fieldnames=("frame_id", "timestamp_ms", "left_state", "right_state",
                "left_open_fraction", "right_open_fraction", "both_event", "both_count", "long_count", "last_duration_ms", "after_read_ms",
                "fpga_left_cycles", "fpga_right_cycles", "fpga_pair_cycles", "validity", "error_reason"))
            writer.writeheader()
        if args.backend in ("fpga", "compare"):
            client = client_factory() if client_factory is not None else (FPGAClient(args.port, baud=stats["baud"]) if pretrained or args.baud else FPGAClient(args.port))
            status = client.ping()
            if not status.get("binocular_timing"):
                raise RuntimeError("FPGA firmware does not support the existing binocular timer")
            if experiment and status.get("method_id") == 4:
                raise RuntimeError("TinyEye firmware required; pretrained firmware cannot run research encoders")
            loaded = client.load_model(model)
            if experiment:
                if not loaded.get("model_loaded") or client.read_model() != model_image(model.params):
                    raise RuntimeError("TinyEye model readback mismatch")
                session.hardware(loaded)
            client.configure(0, 0)
            client.reset_blinks()
        if experiment:
            from experiment import ComparisonPipeline
            pipeline = ComparisonPipeline(model, args.backend, client)
        else:
            pipeline = EyePipeline(model, args.workers, client, args.verify)
            if pipeline.batch_eyes:
                stats["workers"] = 1
                stats["eye_inference"] = "paired_batch"
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        capture_args = (pending, errors, stop, stopped, args.camera)
        if session:
            capture_args += (session.capture,)
        camera_worker = Thread(target=capture_frames, args=capture_args, daemon=True)
        camera_worker.start()
        started = last_capture = time.monotonic()
        previous_ms = -1
        while not args.seconds or time.monotonic() - started < args.seconds:
            if stopped.is_set():
                raise RuntimeError(errors.get_nowait() if not errors.empty() else "Camera stopped")
            try:
                frame_id, captured_ns, frame = pending.get(timeout=0.05)
            except Empty:
                if time.monotonic() - last_capture > 5:
                    raise RuntimeError("Camera stalled for five seconds")
                if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q"), 27):
                    break
                continue
            last_capture = time.monotonic()
            captured_ms = captured_ns // 1_000_000
            if captured_ms <= previous_ms:
                continue
            previous_ms = captured_ms
            rois, boxes = {}, {}
            if time.monotonic_ns() - captured_ns <= MAXIMUM_GAP_MS * 1_000_000:
                landmarks = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), captured_ms)
                if landmarks.face_landmarks:
                    for side in SIDES:
                        roi, box = extract_eye_roi(frame, landmarks.face_landmarks[0], EYE_CORNERS[side])
                        if roi is not None:
                            if pretrained:
                                # Keep original BGR data; never reconstruct color from the legacy gray thumbnail.
                                x0, y0, x1, y1 = box
                                roi = frame[y0:y1, x0:x1].copy()
                            rois[side], boxes[side] = roi, box
            # Landmark processing itself can consume the age budget.
            if time.monotonic_ns() - captured_ns > MAXIMUM_GAP_MS * 1_000_000:
                rois.clear()
            inference_started = time.perf_counter_ns()
            try:
                result = pipeline.predict(frame_id, captured_ns, rois)
            except Exception as exc:
                if session:
                    names = ("cpu", "fpga") if args.backend == "compare" else (args.backend,)
                    failed = PairResult(frame_id, captured_ns, {}, (time.perf_counter_ns()-inference_started)/1e6,
                        backends={name: dict(logits=None, error=str(exc)) for name in names})
                    session.record(failed, [None, None], time.monotonic_ns(), error=str(exc))
                raise
            if stopped.is_set():
                if session:
                    session.record(result, [None, None], time.monotonic_ns(), error="Camera stopped while processing the frame")
                raise RuntimeError("Camera stopped while processing the frame")
            try:
                timing, scores, now_ns = synchronize_timing(timer, result, client)
            except Exception as exc:
                if session:
                    session.record(result, [None, None], time.monotonic_ns(), error=str(exc))
                raise
            age_ms = (now_ns - captured_ns) / 1e6
            states = ["UNAVAILABLE" if scores[index] is None or not timing.valid_mask & (1 << index)
                      else "CLOSED" if timing.closed_mask & (1 << index) else "OPEN" for index in range(2)]
            stats["frames"] += 1
            stats["valid_eye_pairs"] += all(score is not None for score in scores) and timing.valid_mask == 3
            if session:
                session.record(result, scores, now_ns)
            stats["skipped_capture_frames"] += max(0, frame_id - previous_frame_id - 1)
            previous_frame_id = frame_id
            stats["pair_processing_sum_ms"] += result.processing_ms
            stats["pair_processing_max_ms"] = max(stats["pair_processing_max_ms"], result.processing_ms)
            stats["after_read_sum_ms"] += age_ms
            stats["after_read_max_ms"] = max(stats["after_read_max_ms"], age_ms)
            if result.hardware_cycles is not None:
                stats["hardware_cnn_pairs"] += 1
                stats["last_hardware_cycles"] = dict(zip(("left", "right", "pair"), result.hardware_cycles))
            if frame_origin_ms is None:
                frame_origin_ms = captured_ms
            if writer:
                writer.writerow(dict(frame_id=frame_id, timestamp_ms=captured_ms - frame_origin_ms,
                    left_state=states[0], right_state=states[1], left_open_fraction="", right_open_fraction="",
                    validity="stale" if age_ms > MAXIMUM_GAP_MS else "valid" if all(s is not None for s in scores) else "invalid",
                    error_reason="expired_at_acceptance" if age_ms > MAXIMUM_GAP_MS else "" if all(s is not None for s in scores) else "incomplete_eye_observation",
                    both_event=int(timing.event and all(score is not None for score in scores)), both_count=timing.count, long_count=timing.long_count,
                    last_duration_ms=timing.last_duration_ms, after_read_ms=age_ms,
                    **dict(zip(("fpga_left_cycles", "fpga_right_cycles", "fpga_pair_cycles"), result.hardware_cycles or ("", "", "")))))
            display = frame.copy()
            for index, side in enumerate(SIDES):
                text(display, f"{side.upper()}: {states[index]}", 10, 55 + index * 25)
                if side in boxes:
                    x0, y0, x1, y1 = boxes[side]
                    cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 220), 1)
            title = f"{args.backend.upper()} {model.record['architecture']}" if model.ready else "MODEL NOT TRAINED | ROI preview only"
            text(display, title, 10, 25)
            totals = (str(timing.count), str(timing.long_count)) if stats["valid_eye_pairs"] else ("--", "--")
            last_duration = str(timing.last_duration_ms) if timing.count else "--"
            text(display, f"BOTH-eye cycles: {totals[0]} | long: {totals[1]} | last: {last_duration} ms", 10, 110)
            text(display, f"Frame {frame_id} | Pair processing: {result.processing_ms:.1f} ms | age: {age_ms:.1f} ms", 10, 135)
            current = [str(value) if scores[index] is not None else "--"
                       for index, value in enumerate((timing.left_current_ms, timing.right_current_ms))]
            text(display, f"Current closure L/R: {current[0]}/{current[1]} ms", 10, 160)
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        if experiment:
            stats["interrupted"] = True
            exit_code = 130
    except Exception as exc:
        stats["error"] = str(exc)
        print(f"LIVE ERROR: {exc}", flush=True)
        exit_code = 1
    finally:
        stop.set()
        if camera_worker:
            camera_worker.join(timeout=2)
        if pipeline:
            pipeline.close()
        if frame_stream:
            frame_stream.close()
        if landmarker:
            landmarker.close()
        if client:
            try:
                client.invalidate(0)
            except Exception as exc:
                print(f"FPGA shutdown error: {exc}", flush=True)
                exit_code = 1
            finally:
                client.close()
        if session:
            session.finish("interrupted" if stats.get("interrupted") else "complete" if exit_code == 0 else "failed", stats.get("error"))
        cv2.destroyAllWindows()
        stats["closures"] = {"total": timer.count if stats["valid_eye_pairs"] else None,
                             "long": timer.long_count if stats["valid_eye_pairs"] else None,
                             "last_duration_ms": timer.last_duration_ms if timer.count else None}
        for prefix in ("pair_processing", "after_read"):
            stats[prefix + "_mean_ms"] = stats.pop(prefix + "_sum_ms") / stats["frames"] if stats["frames"] else None
        stats["status"] = "PASS" if exit_code == 0 else "FAIL"
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_name(args.report.name + ".tmp")
            temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.report)
        except OSError as exc:
            print(f"REPORT ERROR: {exc}", flush=True)
            exit_code = 1
        print("LIVE SUMMARY " + json.dumps(stats), flush=True)
    return exit_code
