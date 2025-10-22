import argparse
import logging
import sys

from autogamen.game.reinforcement_learning import (
    generate_html_report,
    run_reinforcement_learning,
)

parser = argparse.ArgumentParser(description="Train TD-GAMMON using reinforcement learning")
parser.add_argument("--games", help="Number of games to play", default=1000, type=int)
parser.add_argument("--checkpoint", help="Save model after this many training games", default=200, type=int)
parser.add_argument("--alpha", help="Learning rate (default: 0.1)", default=0.1, type=float)
parser.add_argument("--profile", help="Run Python profiler", action='store_true')
parser.add_argument("--exhibition", help="Run exhibition matches during checkpoints", action='store_true')
parser.add_argument("--verbosity", help="Logging verbosity (debug/info/warning)", default="info")
args = parser.parse_args()


if __name__ == "__main__":
    # Housekeeping: log levels
    numeric_level = getattr(logging, args.verbosity.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.verbosity}')
    logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

    print("=" * 80)
    print("TD-GAMMON Training")
    print("=" * 80)
    print(f"Games: {args.games}")
    print(f"Checkpoint interval: {args.checkpoint}")
    print(f"Learning rate: {args.alpha}")
    print(f"Exhibition matches: {args.exhibition}")
    print("=" * 80)

    # Run training
    timestamp, metrics, net = run_reinforcement_learning(
        games=args.games,
        checkpoint_interval=args.checkpoint,
        alpha=args.alpha,
        profile=args.profile,
        exhibition=args.exhibition,
    )

    # Generate HTML report
    print("\nGenerating HTML report...")
    report_path = generate_html_report(timestamp, metrics, args.games)
    print(f"Report saved to: {report_path}")
    print("\nTraining complete!")

    sys.exit(0)
