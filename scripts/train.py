import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_mappo import train, args if "args" in locals() else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SwarmRL 3D MAPPO Training Engine")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--comm-radius", type=float, default=50.0)
    parser.add_argument("--survivors", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    from train_mappo import train
    train(args)
