# ruff: noqa: PLC0415
import functools
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click


def with_log_level[F: Callable[..., Any]](f: F) -> F:
    """decorator to add --log-level option and initialize logger"""
    @click.option(
        "--log-level",
        type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
        default="info",
        help="logging level (default: info)",
    )
    @functools.wraps(f)
    def wrapper(*args: Any, log_level: str, **kwargs: Any) -> Any:
        level_name = log_level.upper()
        level = getattr(logging, level_name)
        logging.basicConfig(level=level, format="%(asctime)s: %(message)s")
        return f(*args, **kwargs)
    return wrapper  # type: ignore


@click.group()
def cli() -> None:
    """autogamen: backgammon AI training and evaluation.

    workflow overview:
      1. train      → train AI models using various approaches
      2. eval       → evaluate model performance
      3. play       → run games with trained models
    """


@cli.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("tmp/annotations"),
    help="directory containing annotation data (default: tmp/annotations)",
)
@click.option(
    "--promote",
    is_flag=True,
    help="promote trained model to models/ directory",
)
@with_log_level
def train(annotations_dir: Path, promote: bool) -> None:
    """train AI model on annotations.

    trains a model to play backgammon based on training data.
    saves checkpoint to tmp/train/ with timestamp in filename.

    examples:
      autogamen train                           # train and save to tmp/train/
      autogamen train --promote                 # train and activate immediately
    """
    # TODO: Implement training logic
    print("Training not yet implemented")


@cli.command()
@click.argument("checkpoint")
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("tmp/annotations"),
    help="directory containing annotations (default: tmp/annotations)",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="limit number of games to evaluate (default: no limit)",
)
@with_log_level
def eval(
    checkpoint: str,
    annotations_dir: Path,
    limit: int | None,
) -> None:
    """evaluate a checkpoint against annotations.

    measures how well a model performs by comparing predictions to stored
    annotations or by playing evaluation games.

    checkpoint can be:
    - path to .pt file (local model checkpoint)
    - model name

    results are written to tmp/eval/{timestamp}_eval.json for later inspection.

    examples:
      autogamen eval models/model.pt                     # evaluate active model
      autogamen eval tmp/train/model_abc123.pt           # specific checkpoint
      autogamen eval models/model.pt --limit 100         # quick eval on subset
    """
    # TODO: Implement evaluation logic
    print("Evaluation not yet implemented")


@cli.command()
@click.argument("white", default="Bozo")
@click.argument("black", default="Bozo")
@click.option(
    "--match-points",
    help="Points for the match",
    default=10,
    type=int,
)
@click.option(
    "--matches",
    help="Number of matches to play",
    default=1,
    type=int,
)
@click.option(
    "--parallelism",
    help="Number processes to use",
    default=1,
    type=int,
)
@click.option(
    "--seed",
    help="Random seed, for deterministic testing",
    type=int,
)
@with_log_level
def headless(
    white: str,
    black: str,
    match_points: int,
    matches: int,
    parallelism: int,
    seed: int | None,
) -> None:
    """run headless game matches.

    runs backgammon matches without UI for benchmarking and training.

    examples:
      autogamen headless Bozo Delta                        # single match
      autogamen headless Bozo Delta --matches 100          # 100 matches
      autogamen headless Bozo Delta --parallelism 4        # parallel execution
    """
    from autogamen.game.headless import run_headless_matches
    run_headless_matches(white, black, match_points, matches, parallelism, seed)


@cli.command()
@click.argument("opponent", default="Delta")
@click.option(
    "--points",
    help="Point limit for the match to play",
    default=3,
    type=int,
)
@with_log_level
def ui(opponent: str, points: int) -> None:
    """run interactive UI match against AI.

    launches a graphical interface to play backgammon against an AI opponent.

    examples:
      autogamen ui Delta                 # play against Delta
      autogamen ui Bozo --points 5       # 5-point match
    """
    from autogamen.game.ui_match import run_ui_match
    run_ui_match(opponent, points)


@cli.command(name="neuro-evolution")
@click.option(
    "--generations",
    help="Number of generations to run",
    default=10,
    type=int,
)
@click.option(
    "--population",
    help="Number of players in each generation",
    default=10,
    type=int,
)
@click.option(
    "--parallelism",
    help="Number processes to use",
    default=1,
    type=int,
)
@click.option(
    "--mutation",
    help="Mutation rate",
    default=0.05,
    type=float,
)
@click.option(
    "--crossover",
    help="Percent of generation comprised of children",
    default=0.8,
    type=float,
)
@with_log_level
def neuro_evolution(
    generations: int,
    population: int,
    parallelism: int,
    mutation: float,
    crossover: float,
) -> None:
    """train using neuro-evolution.

    evolves neural network players through genetic algorithms.

    examples:
      autogamen neuro-evolution                                    # default settings
      autogamen neuro-evolution --generations 50 --population 20   # custom parameters
    """
    from autogamen.game.neuro_evolution import run_neuro_evolution
    run_neuro_evolution(generations, population, parallelism, mutation, crossover)


@cli.command(name="reinforcement-learning")
@click.option(
    "--games",
    help="Number of games to play",
    default=10,
    type=int,
)
@click.option(
    "--checkpoint",
    help="Save model after this many training games",
    default=1000,
    type=int,
)
@click.option(
    "--alpha",
    help="Learning rate",
    default=0.1,
    type=float,
)
@click.option(
    "--profile",
    help="Run Python profiler",
    is_flag=True,
)
@click.option(
    "--exhibition",
    help="Run exhibition matches during checkpoints",
    is_flag=True,
)
@with_log_level
def reinforcement_learning(
    games: int,
    checkpoint: int,
    alpha: float,
    profile: bool,
    exhibition: bool,
) -> None:
    """train using reinforcement learning.

    trains a neural network using temporal difference learning (TD-lambda).

    examples:
      autogamen reinforcement-learning --games 10000              # 10k games
      autogamen reinforcement-learning --exhibition               # with eval matches
    """
    from autogamen.game.reinforcement_learning import run_reinforcement_learning
    run_reinforcement_learning(games, checkpoint, alpha, profile, exhibition)


if __name__ == "__main__":
    print("✘ don't run cli.py directly - use 'uv run autogamen' instead", file=sys.stderr)
    sys.exit(1)
