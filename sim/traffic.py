"""
traffic.py - Traffic generation and per-UE queueing with delay accounting.
==========================================================================

The baseline generates per-UE UDP traffic with iPerf at three rate levels.
We reproduce that with a Poisson packet arrival process at a fixed target rate
per UE, feeding a FIFO byte queue.

Delay is the quantity the whole QoS side of the reward depends on, so it is
measured properly rather than approximated: every packet carries its arrival
timestamp, and the delay of a served packet is the time it spent waiting.
Partially served packets keep their original timestamp, which is what a real
RLC buffer does.

Classes
-------
``TrafficSource``  Poisson packet generator at a target bit rate
``UeQueue``        FIFO byte queue that records arrival and departure delays
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import numpy as np

from .params import UDP_PACKET_BYTES


# ---------------------------------------------------------------------------
# Traffic generation
# ---------------------------------------------------------------------------


@dataclass
class TrafficSource:
    """Poisson packet arrivals at a fixed mean bit rate.

    Each UE is assigned one source with a rate drawn from its traffic level's
    range at construction, so UEs within a level are heterogeneous but stable
    over an episode - which is how iPerf behaves.
    """

    rate_mbps: float
    packet_bytes: int = UDP_PACKET_BYTES

    def arrivals(self, duration_ms: float, rng: np.random.Generator) -> int:
        """Number of packets that arrive in a window of ``duration_ms``.

        Mean packets in the window = rate * duration / packet size.
        """
        bits = self.rate_mbps * 1e6 * (duration_ms / 1000.0)
        mean_packets = bits / (self.packet_bytes * 8.0)
        return int(rng.poisson(mean_packets))


def assign_traffic_rates(
    n_ue: int, level_range: Tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    """Draw one fixed rate per UE, uniform within the level's range."""
    lo, hi = level_range
    return rng.uniform(lo, hi, size=n_ue)


# ---------------------------------------------------------------------------
# Queueing
# ---------------------------------------------------------------------------


@dataclass
class UeQueue:
    """FIFO byte queue for one UE, with head-of-line delay tracking.

    Packets are stored as ``(arrival_time_ms, remaining_bytes)``. Service
    consumes bytes from the head. A packet is only counted as delivered - and
    only then contributes a delay sample - once its last byte has been served.

    ``max_bytes`` bounds the buffer so that an overloaded UE drops rather than
    growing without limit, which keeps the delay statistic meaningful under
    heavy traffic.
    """

    max_bytes: int = 2_000_000
    _packets: Deque[List[float]] = field(default_factory=deque, repr=False)
    _queued_bytes: int = 0

    # Statistics accumulated since the last call to ``collect``.
    _served_bytes: int = 0
    _dropped_bytes: int = 0
    _delay_samples: List[float] = field(default_factory=list, repr=False)
    _arrived_bytes: int = 0

    # ---- producer side ----------------------------------------------------

    def enqueue(self, n_packets: int, packet_bytes: int, now_ms: float) -> None:
        """Append ``n_packets`` arriving at time ``now_ms``."""
        for _ in range(n_packets):
            if self._queued_bytes + packet_bytes > self.max_bytes:
                self._dropped_bytes += packet_bytes
                continue
            self._packets.append([now_ms, float(packet_bytes)])
            self._queued_bytes += packet_bytes
            self._arrived_bytes += packet_bytes

    # ---- consumer side ----------------------------------------------------

    def serve(self, n_bytes: float, now_ms: float) -> int:
        """Serve up to ``n_bytes`` from the head of the queue.

        Returns the number of bytes actually served, which is less than
        requested when the queue empties first.
        """
        budget = float(n_bytes)
        served = 0
        while budget > 0 and self._packets:
            head = self._packets[0]
            take = min(budget, head[1])
            head[1] -= take
            budget -= take
            served += take
            self._queued_bytes -= take
            if head[1] <= 1e-9:
                # Last byte of this packet: record its end-to-end queueing delay.
                self._delay_samples.append(max(0.0, now_ms - head[0]))
                self._packets.popleft()
        self._served_bytes += int(served)
        return int(served)

    # ---- observation side -------------------------------------------------

    @property
    def queued_bytes(self) -> int:
        return int(self._queued_bytes)

    def head_of_line_delay_ms(self, now_ms: float) -> float:
        """Age of the oldest byte still waiting, or 0 when the queue is empty.

        This matters under sleep: if nothing is served during a window there
        are no completion samples, but the buffer is still ageing and the QoS
        penalty must see that.
        """
        if not self._packets:
            return 0.0
        return max(0.0, now_ms - self._packets[0][0])

    def collect(self, now_ms: float) -> dict:
        """Return the statistics for the window just finished and reset them.

        ``delay_ms`` is the mean delay of packets completed in the window; when
        none completed it falls back to the head-of-line delay so that a
        starved UE is correctly seen as violating its delay target.
        """
        if self._delay_samples:
            delay = float(np.mean(self._delay_samples))
        else:
            delay = self.head_of_line_delay_ms(now_ms)

        stats = {
            "served_bytes": self._served_bytes,
            "arrived_bytes": self._arrived_bytes,
            "dropped_bytes": self._dropped_bytes,
            "queued_bytes": self.queued_bytes,
            "delay_ms": delay,
            "n_completed": len(self._delay_samples),
        }
        self._served_bytes = 0
        self._arrived_bytes = 0
        self._dropped_bytes = 0
        self._delay_samples = []
        return stats

    def reset(self) -> None:
        self._packets.clear()
        self._queued_bytes = 0
        self._served_bytes = 0
        self._arrived_bytes = 0
        self._dropped_bytes = 0
        self._delay_samples = []
