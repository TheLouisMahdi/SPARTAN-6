"""Bounded eye stages. The transport belongs exclusively to the inference thread.

No GUI, device discovery or device opening occurs on import. Temporal state is
owned by the result thread; hardware-timer diagnostics remain in live_pipeline.
"""
from collections import Counter
import hashlib
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from types import MappingProxyType
import time
import numpy as np
from blink_timing import BinocularTimer, MAXIMUM_GAP_MS

SIDES = ('left', 'right')


def process_rss_bytes():
    """Optional Windows working-set observation, not a CPU utilization estimate."""
    import ctypes as c
    import sys
    if sys.platform!='win32': return None
    class Memory(c.Structure):
        _fields_=[('cb',c.c_uint32),('faults',c.c_uint32)]+[(name,c.c_size_t) for name in ('peak','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile')]
    memory=Memory();memory.cb=c.sizeof(memory)
    api=c.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    api.argtypes=[c.c_void_p,c.POINTER(Memory),c.c_uint32];api.restype=c.c_int
    return memory.working if api(c.c_void_p(-1),c.byref(memory),memory.cb) else None


@dataclass(frozen=True)
class PreparedFrame:
    frame_id: int
    capture_ns: int
    frame: np.ndarray
    rois: object
    boxes: object
    prep_start_ns: int
    prep_end_ns: int


@dataclass(frozen=True)
class InferenceResult:
    prepared: PreparedFrame
    hardware: object
    start_ns: int
    end_ns: int
    reason: str | None = None


@dataclass(frozen=True)
class PublishedFrame:
    inference: InferenceResult
    eyes: object
    timing: object
    result_start_ns: int
    result_end_ns: int
    reason: str | None


def offer_latest(queue, item):
    """One producer per queue. Return the actual evicted object, if any."""
    try:
        queue.put_nowait(item)
        return None
    except Full:
        try:
            old = queue.get_nowait()
        except Empty:
            old = None
        queue.put_nowait(item)
        return old


def readonly(array):
    value = np.array(array, copy=True, order='C')
    value.flags.writeable = False
    return value


class StreamingPipeline:
    """Source/prepare factories run on preparation owner; client factory on UART owner.

    Source yields original (frame_id, capture_ns, frame). Prepare returns (rois,
    boxes). Each factory may return an object with close(). No shared MediaPipe
    or serial instance is passed between owners. Sequential uses the identical
    stages with a completion barrier before accepting the next source frame.
    """
    def __init__(self, source_factory, prepare_factory, client_factory, model,
                 mode='overlap', verify=False, emit=None, clock=time.perf_counter_ns, startup_timeout=5, prepared_capacity=1):
        if mode not in ('sequential', 'overlap'):
            raise ValueError('Unknown pipeline mode')
        if prepared_capacity not in (1,2,3,4):
            raise ValueError('Prepared queue must remain bounded to 1..4 frames')
        self.source_factory, self.prepare_factory = source_factory, prepare_factory
        self.client_factory, self.model = client_factory, model
        self.pretrained = model.method == 'omz_bgr32'
        self.mode, self.verify, self.clock = mode, verify, clock
        self.startup_timeout = startup_timeout
        self.prepared, self.inferred, self.display = Queue(prepared_capacity), Queue(2), Queue(1)
        self.errors = Queue(1)
        self.stop, self.abort = Event(), Event()
        self.prep_done, self.fpga_done, self.done, self.ready = Event(), Event(), Event(), Event()
        self.step = Event(); self.step.set()
        self.counts, self.lock, self.emit = Counter(), Lock(), emit
        self.threads = []
        self.identity = None
        self.first_capture_ns = None
        self.integrity = {}

    def record(self, kind, **fields):
        with self.lock:
            if kind=='capture' and self.first_capture_ns is None: self.first_capture_ns=fields['capture_ns']
            self.counts[kind] += 1
            if self.emit:
                self.emit(dict(kind=kind, at_ns=self.clock(), **fields))

    def _fail(self, stage, exc):
        try: self.errors.put_nowait((stage, exc))
        except Full: pass
        self.abort.set(); self.stop.set(); self.step.set()
        try: self.record('worker_failure', stage=stage, error=str(exc), error_type=type(exc).__name__)
        except Exception: pass

    def _put(self, queue, item):
        while not self.abort.is_set():
            try:
                queue.put(item, timeout=.05)
                return True
            except Full: pass
        self.record('cancelled_result', frame_id=item.prepared.frame_id)
        return False

    def start(self):
        if self.threads: raise RuntimeError('Pipeline already started')
        for name, target in [('fpga', self._fpga), ('result', self._result), ('prepare', self._prepare)]:
            worker = Thread(name='eye-'+name, target=target, daemon=True)
            self.threads.append(worker); worker.start()
        return self

    def _prepare(self):
        prepare = source = None
        previous_id = previous_ns = 0
        try:
            deadline = time.monotonic()+self.startup_timeout
            while not self.ready.wait(.05):
                if self.stop.is_set(): return
                if time.monotonic()>=deadline: raise TimeoutError('FPGA startup did not complete')
            if self.abort.is_set(): return
            prepare = self.prepare_factory()
            source = self.source_factory(self.stop)
            iterator = iter(source)
            while not self.stop.is_set():
                if self.mode == 'sequential':
                    while not self.step.wait(.05):
                        if self.stop.is_set(): return
                    if self.stop.is_set(): break
                try: frame_id, capture_ns, frame = next(iterator)
                except StopIteration: break
                if frame_id <= previous_id or capture_ns <= previous_ns:
                    raise ValueError('duplicate_frame or out_of_order_frame/capture timestamp')
                prior_ns = previous_ns
                previous_id, previous_ns = frame_id, capture_ns
                if prior_ns and capture_ns//1000000 == prior_ns//1000000:
                    self.record('dropped_timestamp', frame_id=frame_id); continue
                start = self.clock()
                owned = readonly(frame)
                rois, boxes = prepare(owned,capture_ns)
                rois = MappingProxyType({s: readonly(roi) for s,roi in rois.items() if s in SIDES and roi is not None})
                boxes = MappingProxyType(dict(boxes))
                end = self.clock()
                item = PreparedFrame(frame_id,capture_ns,owned,rois,boxes,start,end)
                self.record('prepared',frame_id=frame_id,capture_ns=capture_ns,prep_start_ns=start,prep_end_ns=end,valid_eyes=len(rois),
                    input_sha256={s:hashlib.sha256((roi.transpose(2,0,1) if roi.ndim==3 else roi).tobytes()).hexdigest() for s,roi in rois.items()})
                self.step.clear()
                old = offer_latest(self.prepared,item)
                if old is not None: self.record('dropped_preparation',frame_id=old.frame_id,replaced_by=frame_id)
        except Exception as exc: self._fail('preparation',exc)
        finally:
            for resource in (source,prepare):
                if resource is not None and hasattr(resource,'close'):
                    try: resource.close()
                    except Exception as exc: self._fail('preparation_close',exc)
            self.prep_done.set()

    def _fpga(self):
        client = None
        try:
            client = self.client_factory()
            from fpga_host import METHOD_IDS, model_image
            method_id = 4 if self.pretrained else METHOD_IDS[self.model.method]
            self.identity = client.load_model(self.model)
            if (self.identity.get('method_id') != method_id or not self.identity.get('model_loaded') or
                self.identity.get('fingerprint') != int(self.model.checkpoint_sha256[:8],16)):
                raise RuntimeError('Invalid model fingerprint')
            if not self.pretrained and client.read_model() != model_image(self.model.params):
                raise RuntimeError('TinyEye coefficient readback mismatch')
            client.configure(0,0)
            self.record('firmware',identity=self.identity)
            self.ready.set()
            while not self.abort.is_set():
                try: item = self.prepared.get(timeout=.05)
                except Empty:
                    if self.prep_done.is_set(): break
                    continue
                start = self.clock(); hardware = None
                reason = ('stale_input' if not 0<=start-item.capture_ns<=MAXIMUM_GAP_MS*1000000 else
                          'missing_eye' if len(item.rois)!=2 else None)
                if reason:
                    client.invalidate(item.frame_id)
                else:
                    self.record('submitted',frame_id=item.frame_id)
                    if self.pretrained:
                        pair = [item.rois[side] for side in SIDES]
                        if getattr(client, 'supports_capture_ns', False):
                            hardware = client.infer_pretrained(*pair, item.frame_id, capture_ns=item.capture_ns)
                        else:
                            hardware = client.infer_pretrained(*pair, item.frame_id)
                    else:
                        from encoders import METHODS
                        codes = [METHODS[self.model.method].encode(item.rois[s]) for s in SIDES]
                        if getattr(client,'supports_capture_ns',False):
                            hardware = client.infer(self.model.method,*codes,item.frame_id,capture_ns=item.capture_ns)
                        else:
                            hardware = client.infer(self.model.method,*codes,item.frame_id,(item.capture_ns//1000000)&0xffffffff)
                    if hardware.frame_id != item.frame_id or hardware.method_id != method_id:
                        raise RuntimeError('Frame ID / method mismatch')
                    self.record('fpga_completed',frame_id=item.frame_id)
                end = self.clock()
                integrity = dict(crc_errors=getattr(getattr(client,'reader',None),'bad_packets',0),
                                 timeouts=getattr(client,'timeout_count',0),sequence_errors=getattr(client,'unmatched_responses',0))
                self.integrity = integrity
                if any(integrity.values()): raise RuntimeError('UART integrity failure: '+str(integrity))
                self.record('inference',frame_id=item.frame_id,fpga_start_ns=start,fpga_end_ns=end,reason=reason,
                            cycles=None if hardware is None else [hardware.left_cycles,hardware.right_cycles,hardware.pair_cycles],
                            uart_roundtrip_ms=None if hardware is None else hardware.roundtrip_ms,
                            uart_upload_ms=None if hardware is None else hardware.upload_ms,
                            kernel_envelopes=getattr(hardware,'kernel_envelopes',None))
                if not self._put(self.inferred,InferenceResult(item,hardware,start,end,reason)): break
        except Exception as exc: self._fail('fpga',exc)
        finally:
            self.ready.set()
            if client is not None:
                try: client.close()
                except Exception as exc: self._fail('fpga_close',exc)
            self.fpga_done.set()

    def _result(self):
        timer = BinocularTimer(); last_id = last_ns = 0
        try:
            while not self.abort.is_set():
                try: result = self.inferred.get(timeout=.05)
                except Empty:
                    if self.fpga_done.is_set(): break
                    continue
                item = result.prepared; start = self.clock()
                if item.frame_id<=last_id or item.capture_ns<=last_ns:
                    raise RuntimeError('duplicate_frame or out_of_order_frame at result stage')
                if last_id and item.frame_id != last_id+1:
                    self.record('missing_frame',frame_id=item.frame_id,previous_id=last_id)
                    timer.invalidate()
                last_id,last_ns=item.frame_id,item.capture_ns
                reason = result.reason
                if not 0<=start-item.capture_ns<=MAXIMUM_GAP_MS*1000000: reason='stale_result'
                eyes = {s:dict(state='UNAVAILABLE',logits=None) for s in SIDES}
                hardware = result.hardware
                if hardware is not None:
                    for side,logits in zip(SIDES,(hardware.left_logits,hardware.right_logits)):
                        if self.verify:
                            if self.pretrained:
                                expected=self.model.predict(item.rois[side],fixed=True)
                            else:
                                from encoders import METHODS
                                from integer_model import infer
                                codes=METHODS[self.model.method].encode(item.rois[side])
                                pixels=codes if self.model.method=='grayscale8' else codes*np.uint8(85)
                                expected=infer(pixels[None,None],self.model.params)[0]
                            if not np.array_equal(logits,expected): raise RuntimeError('CPU/FPGA exact logit mismatch')
                        if reason is None:
                            eyes[side]=dict(logits=list(map(int,logits)),state='OPEN' if logits[0]>logits[1] else 'CLOSED' if logits[1]>logits[0] else 'UNAVAILABLE')
                # Verification is optional and can itself make a result stale.
                if not 0<=self.clock()-item.capture_ns<=MAXIMUM_GAP_MS*1000000:
                    reason='stale_result'; eyes={s:dict(state='UNAVAILABLE',logits=None) for s in SIDES}
                scores=[None if eyes[s]['state']=='UNAVAILABLE' else float(eyes[s]['state']=='CLOSED') for s in SIDES]
                timing=timer.update(*scores,(item.capture_ns//1000000)&0xffffffff)
                end=self.clock()
                self.record('published',frame_id=item.frame_id,capture_ns=item.capture_ns,result_start_ns=start,result_end_ns=end,
                            reason=reason,eyes=eyes,blink_count=timing.count,blink_event=timing.event,
                            raw_logits=None if hardware is None else [list(hardware.left_logits),list(hardware.right_logits)])
                old=offer_latest(self.display,PublishedFrame(result,MappingProxyType(eyes),timing,start,end,reason))
                if old is not None: self.record('dropped_display',frame_id=old.inference.prepared.frame_id)
                self.step.set()
        except Exception as exc: self._fail('result',exc)
        finally: self.done.set(); self.step.set()

    def close(self, timeout=6):
        self.stop.set(); self.step.set()
        deadline=time.monotonic()+timeout
        for worker in self.threads: worker.join(max(0,deadline-time.monotonic()))
        alive=[w.name for w in self.threads if w.is_alive()]
        if alive: self.abort.set(); raise RuntimeError('Workers did not stop: '+str(alive))
        self.record('shutdown',clean=True)

    def raise_error(self):
        if not self.errors.empty():
            stage,exc=self.errors.get_nowait()
            raise RuntimeError(f'{stage}: {exc}') from exc
