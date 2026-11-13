"""Sample resident process memory during candidate training, including native code."""
from __future__ import annotations

import threading

import psutil


class ProcessMemorySampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.process = psutil.Process()
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def start(self):
        self._thread.start()

    def stop(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()

    def record(self):
        return dict(rss_before_bytes=self.before, sampled_peak_rss_bytes=self.peak,
                    rss_increase_bytes=max(0, self.peak - self.before), rss_sample_interval_seconds=self.interval)
