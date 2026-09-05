"""Temporal event logic — turning per-sample scores into events.

Handles the property the problem statement stresses: anomalies unfold at
different time scales. Accidents are over in a second, congestion builds
gradually, a stopped vehicle only becomes anomalous after being stationary.
A small state machine with per-class debounce converts the trigger stream
into confirmed events, closes them after a quiet period, and suppresses
duplicate alerts with a cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import EVENT_CLEAR_SECS, EVENT_COOLDOWN_SECS, EVENT_CONFIRM_SECS


@dataclass
class Event:
    class_name: str
    start_t: float
    last_t: float
    confirmed: bool = False
    confirm_t: float | None = None
    description: str = ""
    max_confidence: float = 0.0


@dataclass
class TemporalAggregator:
    confirm_secs: float = EVENT_CONFIRM_SECS
    clear_secs: float = EVENT_CLEAR_SECS
    cooldown_secs: float = EVENT_COOLDOWN_SECS
    events: list[Event] = field(default_factory=list)
    _last_alert: dict[str, float] = field(default_factory=dict)

    def update(self, t: float, class_name: str, confidence: float, description: str = "") -> list[Event]:
        """Feed one stage-2-confirmed observation; returns newly confirmed events."""
        newly: list[Event] = []

        for ev in self.events:
            if ev.class_name == class_name and not self._closed(ev, t):
                ev.last_t = t
                ev.max_confidence = max(ev.max_confidence, confidence)
                if description:
                    ev.description = description
                if not ev.confirmed and t - ev.start_t >= self.confirm_secs:
                    ev.confirmed = True
                    ev.confirm_t = t
                    newly.append(ev)
                break
        else:
            ev = Event(class_name=class_name, start_t=t, last_t=t,
                       max_confidence=confidence, description=description)
            self.events.append(ev)
            if self.confirm_secs <= 0:
                ev.confirmed = True
                ev.confirm_t = t
                newly.append(ev)

        self._gc(t)
        return newly

    def _closed(self, ev: Event, t: float) -> bool:
        return ev.confirmed and (t - ev.last_t) > self.clear_secs

    def _gc(self, t: float) -> None:
        self.events = [e for e in self.events if not self._closed(e, t)]

    def alert_allowed(self, ev: Event, t: float) -> bool:
        last = self._last_alert.get(ev.class_name, -1e9)
        if t - last >= self.cooldown_secs:
            self._last_alert[ev.class_name] = t
            return True
        return False

    def confirmed_events(self) -> list[Event]:
        return [e for e in self.events if e.confirmed]
