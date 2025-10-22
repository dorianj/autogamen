"""Headless game match runner - migrated from run_headless_game.py"""
import logging
import random
import sys
from collections import Counter
from multiprocessing import Pool

from autogamen.game.match import Match
from autogamen.game.types import Color


def _fmt_percent(p: float) -> str:
    return f"{p:.1%}"


def run_match_args(args: tuple) -> "Match":
    """Wrapper to configure logging in subprocess"""
    verbosity, white_cls, black_cls, point_limit = args
    numeric_level = getattr(logging, verbosity.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {verbosity}')
    logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")
    return run_match(white_cls, black_cls, point_limit)


def run_match(white_player_cls: type, black_player_cls: type, point_limit: int) -> "Match":
    """Run a single match to completion"""
    match = Match([white_player_cls(Color.White), black_player_cls(Color.Black)], point_limit)
    match.start_game()
    while True:
        if match.tick():
            if match.winner is not None:
                game_count = len(match.games)
                logging.info(f"Match ended: {match.winner.color} won with {match.points[match.winner.color]} points in {game_count} games with {match.turn_count} turns")
                for color, wins in Counter(game.winner.color for game in match.games).items():
                    logging.info(f"{color} won {wins} games ({_fmt_percent(wins / game_count)})")
                return match
            else:
                game = match.current_game
                logging.info(f"Game ended: {game.winner.color} won with {game.points} points after {game.turn_number} turns.")
                match.start_game()


def run_headless_matches(
    white: str,
    black: str,
    match_points: int,
    matches: int,
    parallelism: int,
    seed: int | None,
) -> None:
    """Main entry point for headless matches"""
    # Random seed
    if seed:
        random.seed(seed)

    # Initialize players - TODO: make this more flexible
    # For now, just use globals from the main namespace
    # This will need to be refactored to import from a player registry
    players_cls = [globals().get(f"{name}Player") for name in (white, black)]
    if None in players_cls:
        raise ValueError(f"Unknown player type(s): {white}, {black}")

    verbosity = logging.getLevelName(logging.getLogger().getEffectiveLevel()).lower()

    # Run the matches
    with Pool(processes=parallelism) as pool:
        args_list = [(verbosity, players_cls[0], players_cls[1], match_points) for _ in range(matches)]
        completed_matches = pool.map(run_match_args, args_list)

    # Print results
    print(f"{sum(len(match.games) for match in completed_matches)} games finished:")
    win_counts = Counter(match.winner.color for match in completed_matches)
    for player in completed_matches[0].players:
        wins = win_counts[player.color]
        print(f"{type(player).__name__} ({player.color}) won {wins} matches ({_fmt_percent(wins / len(completed_matches))})")

    sys.exit(0)
