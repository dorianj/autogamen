# ruff: noqa: PLC0415
import functools
import logging
import sys
from collections.abc import Callable
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
      2. battle     → run battles between models
      3. play       → run games with trained models
    """




@cli.command()
@click.argument("model1")
@click.argument("model2")
@click.option(
    "--matches",
    type=int,
    default=10,
    help="number of matches to play (default: 10)",
)
@click.option(
    "--match-points",
    type=int,
    default=5,
    help="points per match (default: 5)",
)
@click.option(
    "--parallelism",
    type=int,
    default=1,
    help="number of processes to use (default: 1)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="random seed for deterministic results",
)
@with_log_level
def battle(
    model1: str,
    model2: str,
    matches: int,
    match_points: int,
    parallelism: int,
    seed: int | None,
) -> None:
    """run battles between two models.

    runs multiple matches between two models and reports statistics.

    models can be:
    - path to .torch checkpoint file
    - player class name (e.g., Bozo, Delta)

    examples:
      autogamen battle Bozo Delta                                  # 10 matches
      autogamen battle models/net.torch Bozo --matches 100         # 100 matches
      autogamen battle models/net1.torch models/net2.torch         # two checkpoints
    """
    import random
    from collections import Counter
    from multiprocessing import Pool

    from autogamen.game.game_types import Color
    from autogamen.game.match import Match

    # Random seed
    if seed:
        random.seed(seed)

    # Initialize players - TODO: make this more flexible
    players_cls = [globals().get(f"{name}Player") for name in (model1, model2)]
    if None in players_cls:
        raise ValueError(f"Unknown player type(s): {model1}, {model2}")

    verbosity = logging.getLevelName(logging.getLogger().getEffectiveLevel()).lower()

    def _fmt_percent(p: float) -> str:
        return f"{p:.1%}"

    def run_match_args(args: tuple[Any, ...]) -> Match:
        """Wrapper to configure logging in subprocess"""
        verbosity, white_cls, black_cls, point_limit = args
        numeric_level = getattr(logging, verbosity.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {verbosity}')
        logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

        match = Match([white_cls(Color.White), black_cls(Color.Black)], point_limit)
        while True:
            if match.tick():
                assert match.winner is not None
                game_count = len(match.games)
                logging.info(f"Match ended: {match.winner.color} won with {match.points[match.winner.color]} points in {game_count} games with {match.turn_count} turns")
                winners = [g.winner for g in match.games if g.winner]
                for color, wins in Counter(w.color for w in winners).items():  # type: ignore[attr-defined]
                    logging.info(f"{color} won {wins} games ({_fmt_percent(wins / game_count)})")
                return match

    # Run the matches
    with Pool(processes=parallelism) as pool:
        args_list = [(verbosity, players_cls[0], players_cls[1], match_points) for _ in range(matches)]
        completed_matches = pool.map(run_match_args, args_list)

    # Print results
    print(f"{sum(len(match.games) for match in completed_matches)} games finished:")
    win_counts = Counter(match.winner.color for match in completed_matches if match.winner)
    for player in completed_matches[0].players:
        wins = win_counts[player.color]
        print(f"{type(player).__name__} ({player.color}) won {wins} matches ({_fmt_percent(wins / len(completed_matches))})")


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
    import random
    from collections import Counter
    from multiprocessing import Pool

    from autogamen.game.game_types import Color
    from autogamen.game.match import Match

    # Random seed
    if seed:
        random.seed(seed)

    # Initialize players - TODO: make this more flexible
    players_cls = [globals().get(f"{name}Player") for name in (white, black)]
    if None in players_cls:
        raise ValueError(f"Unknown player type(s): {white}, {black}")

    verbosity = logging.getLevelName(logging.getLogger().getEffectiveLevel()).lower()

    def _fmt_percent(p: float) -> str:
        return f"{p:.1%}"

    def run_match_args(args: tuple[Any, ...]) -> Match:
        """Wrapper to configure logging in subprocess"""
        verbosity, white_cls, black_cls, point_limit = args
        numeric_level = getattr(logging, verbosity.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {verbosity}')
        logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

        match = Match([white_cls(Color.White), black_cls(Color.Black)], point_limit)
        while True:
            if match.tick():
                assert match.winner is not None
                game_count = len(match.games)
                logging.info(f"Match ended: {match.winner.color} won with {match.points[match.winner.color]} points in {game_count} games with {match.turn_count} turns")
                winners = [g.winner for g in match.games if g.winner]
                for color, wins in Counter(w.color for w in winners).items():  # type: ignore[attr-defined]
                    logging.info(f"{color} won {wins} games ({_fmt_percent(wins / game_count)})")
                return match

    # Run the matches
    with Pool(processes=parallelism) as pool:
        args_list = [(verbosity, players_cls[0], players_cls[1], match_points) for _ in range(matches)]
        completed_matches = pool.map(run_match_args, args_list)

    # Print results
    print(f"{sum(len(match.games) for match in completed_matches)} games finished:")
    win_counts = Counter(match.winner.color for match in completed_matches if match.winner)
    for player in completed_matches[0].players:
        wins = win_counts[player.color]
        print(f"{type(player).__name__} ({player.color}) won {wins} matches ({_fmt_percent(wins / len(completed_matches))})")


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
    from autogamen.ai.players import DeltaPlayer
    from autogamen.game.game_types import Color
    from autogamen.game.match import Match
    from autogamen.ui.match_view import MatchView
    from autogamen.ui.ui_player import HumanPlayer

    # TODO: make opponent selection more flexible
    # For now, hardcode DeltaPlayer
    opponent_cls = DeltaPlayer

    human_player = HumanPlayer(Color.White)
    match = Match([human_player, opponent_cls(Color.Black)], points)
    match_view = MatchView(match)
    match_view.create_window()
    human_player.attach(match_view)
    match_view.run()


@cli.command(name="train-ne")
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
      autogamen train-ne                                    # default settings
      autogamen train-ne --generations 50 --population 20   # custom parameters
    """
    from autogamen.game.neuro_evolution import run_neuro_evolution
    run_neuro_evolution(generations, population, parallelism, mutation, crossover)


@cli.command(name="train-rl")
@click.option(
    "--games",
    help="Number of games to play",
    default=10,
    type=int,
)
@click.option(
    "--tag",
    help="Tag for this training run (defaults to timestamp_githash)",
    default=None,
    type=str,
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
@with_log_level
def reinforcement_learning(
    games: int,
    tag: str | None,
    alpha: float,
    profile: bool,
) -> None:
    """train using reinforcement learning.

    trains a neural network using temporal difference learning (TD-lambda).

    examples:
      autogamen train-rl --games 10000              # 10k games
      autogamen train-rl --tag myexp                # custom tag
    """
    from autogamen.game.reinforcement_learning import run_reinforcement_learning
    run_reinforcement_learning(games, tag, alpha, profile)


if __name__ == "__main__":
    print("✘ don't run cli.py directly - use 'uv run autogamen' instead", file=sys.stderr)
    sys.exit(1)
