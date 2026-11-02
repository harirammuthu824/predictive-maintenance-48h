import os
import statistics
import time
import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

ENDPOINTS = ["/", "/health", "/model/info"]

def benchmark(path, runs=20):
    times = []
    url = BASE_URL.rstrip("/") + path

    for _ in range(runs):
        start = time.perf_counter()
        response = requests.get(url, timeout=30)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        times.append(elapsed_ms)

    return {
        "endpoint": path,
        "runs": runs,
        "min_ms": min(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "max_ms": max(times),
    }

if __name__ == "__main__":
    print(f"Benchmark target: {BASE_URL}")
    print("-" * 72)
    for endpoint in ENDPOINTS:
        try:
            result = benchmark(endpoint)
            print(
                f"{result['endpoint']:12} "
                f"min={result['min_ms']:.2f} ms | "
                f"mean={result['mean_ms']:.2f} ms | "
                f"median={result['median_ms']:.2f} ms | "
                f"max={result['max_ms']:.2f} ms"
            )
        except Exception as exc:
            print(f"{endpoint:12} ERROR: {exc}")
