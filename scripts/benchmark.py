import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark import run_benchmark

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SwarmRL Automated Benchmark Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--comm-radius", type=float, default=50.0)
    parser.add_argument("--survivors", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="data/benchmark_results.json")

    args = parser.parse_args()
    run_benchmark(args)
