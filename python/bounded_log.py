"""Ordered bounded logging keeps filesystem stalls outside inference owners."""
from pathlib import Path
from queue import Queue, Full
from threading import Event, Thread
import time

_STOP = object()


class BoundedLog:
    def __init__(self, path=None, *, stream=None, formatter=str, capacity=2048):
        if capacity < 1:
            raise ValueError("log capacity must be positive")
        self.stream = stream if stream is not None else Path(path).open("x", encoding="utf-8", buffering=65536)
        self.queue, self.formatter = Queue(capacity), formatter
        self.error = None
        self.closed = False
        self.worker = Thread(target=self._run, name="eye-log-writer", daemon=True)
        self.worker.start()

    def _run(self):
        try:
            while True:
                record = self.queue.get()
                if record is _STOP:
                    break
                if isinstance(record, tuple) and len(record) == 2 and record[0] is _STOP:
                    self.stream.flush()
                    record[1].set()
                else:
                    self.stream.write(self.formatter(record))
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.stream.close()
            except Exception as exc:
                self.error = self.error or exc

    def write(self, record):
        if self.closed or self.error is not None:
            raise RuntimeError("log is closed or failed") from self.error
        try:
            self.queue.put_nowait(record)
        except Full as exc:
            raise BufferError("bounded log queue full; evidence cannot be discarded") from exc

    def flush(self, timeout=15):
        marker = Event()
        self.write((_STOP, marker))
        if not marker.wait(timeout) or self.error is not None:
            raise RuntimeError("log flush did not complete") from self.error

    def close(self, timeout=15):
        if self.closed:
            if self.worker.is_alive():
                self.worker.join(timeout)
            if self.worker.is_alive():
                raise TimeoutError("log writer did not stop")
            if self.error is not None:
                raise RuntimeError("log writer failed") from self.error
            return
        self.closed = True
        deadline = time.monotonic()+timeout
        while self.worker.is_alive() and self.error is None:
            try:
                self.queue.put(_STOP, timeout=min(.05, max(.001, deadline-time.monotonic())))
                break
            except Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("log writer could not drain its bounded queue")
        self.worker.join(max(0, deadline-time.monotonic()))
        if self.worker.is_alive():
            raise TimeoutError("log writer did not stop")
        if self.error is not None:
            raise RuntimeError("log writer failed") from self.error
