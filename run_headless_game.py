from collections import Counter
import argparse
import logging
import random
import sys

from autogamen.ai.simple import BozoPlayer, DeltaPlayer, RunningPlayer
from autogamen.game.match import Match
from autogamen.game.types import Color

parser = argparse.ArgumentParser()
parser.add_argument("white", default="Bozo")
parser.add_argument("black", default="Bozo")
parser.add_argument("--match_points", help="Points for the match", default=10, type=int)
parser.add_argument("--matches", help="Number of matches to play", default=1, type=int)
parser.add_argument("--parallelism", help="Number processes to use", default=1, type=int)
parser.add_argument("--seed", help="Random seed, for determistic testing", type=int)
parser.add_argument("--verbosity", help="Logging verbosity (debug/info/warning)", default="info")


def _fmt_percent(p):
  return "{0:.1%}".format(p)

def _run_match(white_player_cls, black_player_cls, point_limit):
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

if __name__ == "__main__":
  args = parser.parse_args()

  # Housekeeping: log levels
  numeric_level = getattr(logging, args.verbosity.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
  logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

  # Random seed
  if args.seed:
    random.seed(args.seed)

  # Initialize players
  players_cls = [globals()[f"{name}Player"] for name in (args.white, args.black)]

  # Run the matches.
  matches = [_run_match(*players_cls, args.match_points) for i in range(0, args.matches)]

  # Print results
  print(f"{sum(len(match.games) for match in matches)} games finished:")
  win_counts = Counter(match.winner.color for match in matches)
  for player in matches[0].players:
    wins = win_counts[player.color]
    print(f"{type(player).__name__} ({player.color}) won {wins} matches ({_fmt_percent(wins / len(matches))})")

  sys.exit(0)
