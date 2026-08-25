"""Live-progress pacing shared by the CLI and the dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressPacer:
    """Tracks a seconds-per-unit pace across progress ticks, asymmetrically.

    An exponential moving average (nudged, not replaced, by each
    observation — the approach `tqdm` uses), smoothed asymmetrically:
    ``rising_smoothing`` (0.4) partially adopts a tick implying a slower
    pace than currently tracked, ``falling_smoothing`` (0.2) partially
    adopts one implying a faster pace. Weighting a slowdown more than a
    speedup catches up to a genuine sustained slowdown (e.g. an expanding
    walk-forward's later, bigger-training-window folds) faster than a
    symmetric average would; keeping both partial rather than full (no
    single tick fully overrides the tracked rate) avoids one noisy tick
    (parameter-grid candidates genuinely cost different amounts) swinging
    the estimate on its own.
    """

    rising_smoothing: float = 0.4
    falling_smoothing: float = 0.2
    _rate_seconds_per_unit: float | None = field(default=None, init=False)
    _last_done: int = field(default=0, init=False)
    _last_elapsed: float = field(default=0.0, init=False)

    def update(self, done: int, elapsed: float) -> None:
        """Record a new ``(done, elapsed)`` observation."""
        delta_done = done - self._last_done
        delta_elapsed = elapsed - self._last_elapsed
        if delta_done > 0 and delta_elapsed > 0:
            instantaneous = delta_elapsed / delta_done
            if self._rate_seconds_per_unit is None:
                self._rate_seconds_per_unit = instantaneous
            else:
                smoothing = (
                    self.rising_smoothing
                    if instantaneous >= self._rate_seconds_per_unit
                    else self.falling_smoothing
                )
                self._rate_seconds_per_unit = (
                    smoothing * instantaneous
                    + (1 - smoothing) * self._rate_seconds_per_unit
                )
        self._last_done = done
        self._last_elapsed = elapsed

    def remaining(self, done: int, total: int) -> float | None:
        """Return estimated seconds left, or ``None`` before any pace is known."""
        if self._rate_seconds_per_unit is None:
            return None
        return self._rate_seconds_per_unit * max(0, total - done)


class ProgressReporter:
    """Turns ``on_progress(done, total)`` ticks into a status line.

    Shared by the dashboard's Streamlit progress bar and the CLI's terminal
    progress line — same :class:`ProgressPacer`-based ETA either way, so a
    user gets the same, already-tuned estimate whichever interface they run
    a walk-forward/stress-test/sensitivity from. Each caller renders
    :meth:`text` (and, where relevant, :meth:`fraction`) through its own
    mechanism; this class only turns ticks into words.
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self._started = time.monotonic()
        self._pacer = ProgressPacer()
        self._first_call = True

    def fraction(self, done: int, total: int) -> float:
        """Return the completed fraction, in ``[0, 1]``."""
        return min(1.0, done / total) if total > 0 else 0.0

    def text(self, done: int, total: int) -> str:
        """Return the status text for one ``on_progress(done, total)`` tick.

        A first tick with ``done > 0`` can only mean a checkpoint was
        resumed (a fresh run always starts its first tick at 0) — flagged
        in the text that once, since nothing else surfaces a resume to the
        user otherwise.
        """
        now = time.monotonic() - self._started
        if self._first_call and done > 0 and total > 0:
            text = (
                f"{self.title}: resumed from a previous checkpoint at {done}/{total}…"
            )
        elif total > 0 and done >= total:
            text = f"{self.title}: finishing…"
        else:
            self._pacer.update(done, now)
            remaining = self._pacer.remaining(done, total) if total > 0 else None
            if remaining is not None and remaining >= 1.0:
                text = f"{self.title}: {done}/{total} — ~{remaining:.0f}s remaining"
            elif total > 0:
                # No pace yet, or the estimate ran out while work remains —
                # "~0s remaining" would misleadingly read as "any moment
                # now" instead of "still going, longer than expected".
                text = f"{self.title}: {done}/{total}"
            else:
                text = f"{self.title}: running…"
        self._first_call = False
        return text
