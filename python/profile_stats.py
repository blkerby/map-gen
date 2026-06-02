import contextlib
import threading
import time
from collections import defaultdict

import torch


class ProfileStats:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextlib.contextmanager
    def timer(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    @contextlib.contextmanager
    def cuda_timer(self, name: str, device: torch.device):
        if not self.enabled or device.type != "cuda":
            with self.timer(name):
                yield
            return
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize(device)
            self.add(name, time.perf_counter() - start)

    def add(self, name: str, elapsed_seconds: float):
        if not self.enabled:
            return
        with self.lock:
            self.totals[name] += elapsed_seconds
            self.counts[name] += 1

    def reset(self):
        with self.lock:
            self.totals.clear()
            self.counts.clear()

    def metrics(self):
        with self.lock:
            return {
                f"profile/{name}_ms": elapsed_seconds * 1000.0
                for name, elapsed_seconds in self.totals.items()
            }

    def format(self):
        with self.lock:
            return ",\n".join(
                f"{name}={self.totals[name] * 1000.0:.1f}ms/{self.counts[name]}"
                for name in sorted(self.totals)
            )
