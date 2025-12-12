import time

def estimate_cost(text: str, model_rate_per_1k_char: float = 0.0001) -> float:
    """
    Simple character-based cost estimation.
    """
    return len(text) * (model_rate_per_1k_char / 1000)

class LatencyProfiler:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def get_latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
