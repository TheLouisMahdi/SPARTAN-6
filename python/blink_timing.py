"""Capture-timed binocular closure reference and FPGA transport contract.

MediaPipe coefficients are expression scores, not calibrated eyelid opening.
All durations are observed integer milliseconds, limited by camera sampling.
"""
from dataclasses import dataclass
import math

OPEN_Q16 = 26214
CLOSED_Q16 = 39321
MINIMUM_MS = 30
LONG_MS = 600
SYNC_MS = 150
MAXIMUM_GAP_MS = 250
MASK32 = 0xffffffff


def quantize_score(value):
    """Finite coefficient [0,1] -> unsigned Q16 magnitude; None means unavailable."""
    if value is None or not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return int(value * 65535 + 0.5)


@dataclass(frozen=True)
class TimingResult:
    closed_mask: int
    valid_mask: int
    event: bool
    count: int
    long_count: int
    left_current_ms: int
    right_current_ms: int
    last_left_ms: int
    last_right_ms: int
    last_overlap_ms: int
    last_duration_ms: int


class BinocularTimer:
    """One completed cycle requires observed open eyes, overlapping closures, then both open.

    Closing onsets may differ by <=150ms. Each eye and their overlap must last
    >=30ms. Long closures >=600ms are counted separately, never discarded.
    No fixed frame count or upper closure-duration rejection is used.
    """
    def __init__(self):
        self.count = self.long_count = 0
        self.last_left_ms = self.last_right_ms = self.last_overlap_ms = self.last_duration_ms = 0
        self.invalidate()

    @property
    def last_blink_ms(self):
        return self.last_duration_ms

    def invalidate(self):
        """Cancel continuity and partial cycles; retain completed totals/durations."""
        self._previous_ms = None
        self.closed_mask = self.valid_mask = 0
        self._armed = self._active = self._overlap = False
        self._seen = 0
        self._starts = [0, 0]
        self._ends = [0, 0]
        self._union_start = 0

    def update(self, left, right, timestamp_ms):
        """Two optional [0,1] scores + uint32 capture ms -> exact transport result."""
        scores = [quantize_score(left), quantize_score(right)]
        valid = sum((score is not None) << side for side, score in enumerate(scores))
        return self.update_quantized([score or 0 for score in scores], valid, timestamp_ms)

    def update_quantized(self, scores, valid_mask, timestamp_ms):
        """Two uint16 words, validity bits and modulo-2^32 ms -> FPGA-equivalent result."""
        if not 0 <= timestamp_ms <= MASK32 or valid_mask not in range(4) or len(scores) != 2:
            raise ValueError("Expected two uint16 scores, two validity bits and uint32 milliseconds")
        if any(not 0 <= score <= 65535 for score in scores):
            raise ValueError("Coefficient does not fit uint16")
        t = timestamp_ms
        gap = None if self._previous_ms is None else (t - self._previous_ms) & MASK32
        if gap == 0:
            raise ValueError("Repeated capture timestamp")
        if gap is not None and gap > MAXIMUM_GAP_MS:
            self.invalidate()
        previous_closed, previous_known = self.closed_mask, self.valid_mask
        closed = known = 0
        for side, score in enumerate(scores):
            bit = 1 << side
            if valid_mask & bit:
                if score >= CLOSED_Q16:
                    closed |= bit
                    known |= bit
                elif score <= OPEN_Q16:
                    known |= bit
                else:
                    closed |= previous_closed & bit
                    known |= previous_known & bit
            if closed & bit and (not previous_known & bit or not previous_closed & bit):
                self._starts[side] = t
            if previous_closed & bit and not closed & bit:
                self._ends[side] = t
        self.closed_mask, self.valid_mask, self._previous_ms = closed, known, t
        event = False
        if known != 3:
            self._active = self._armed = self._overlap = False
            self._seen = 0
        elif not closed:
            if self._active and self._overlap:
                durations = [(end - start) & MASK32 for end, start in zip(self._ends, self._starts)]
                starts = [(start - self._union_start) & MASK32 for start in self._starts]
                ends = [(end - self._union_start) & MASK32 for end in self._ends]
                overlap = min(ends) - max(starts)
                if min(durations) >= MINIMUM_MS and overlap >= MINIMUM_MS:
                    self.count = (self.count + 1) & MASK32
                    self.long_count = (self.long_count + int(max(durations) >= LONG_MS)) & MASK32
                    self.last_left_ms, self.last_right_ms = durations
                    self.last_overlap_ms = overlap
                    self.last_duration_ms = (t - self._union_start) & MASK32
                    event = True
            self._active = self._overlap = False
            self._seen = 0
            self._armed = True
        elif self._armed and not self._active:
            self._active = True
            self._seen = closed
            self._union_start = t
            self._overlap = closed == 3
        elif self._active:
            if self._seen & ~previous_closed & closed:
                self._active = self._armed = self._overlap = False
            else:
                self._seen |= closed
                if closed == 3:
                    delta = (self._starts[0] - self._starts[1]) & MASK32
                    self._overlap |= delta <= SYNC_MS or ((-delta) & MASK32) <= SYNC_MS
        current = [((t - self._starts[side]) & MASK32) if closed & known & (1 << side) else 0 for side in range(2)]
        return TimingResult(closed, known, event, self.count, self.long_count, *current,
                            self.last_left_ms, self.last_right_ms, self.last_overlap_ms, self.last_duration_ms)
