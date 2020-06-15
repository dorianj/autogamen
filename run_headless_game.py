from collections import Counter
import argparse
import random
import sys

from autogamen.ai.simple import BozoPlayer, DeltaPlayer, RunningPlayer
from autogamen.game.match import Match
from autogamen.game.types import Color

parser = argparse.ArgumentParser()
parser.add_argument("white", default="Bozo")
parser.add_argument("black", default="Bozo")
parser.add_argument("--match_points", help="Points for the match", default=10)
parser.add_argument("--matches", help="Number of matches to play", default=1)
print(parser.parse_args())


def _run_match(white_player_cls, black_player_cls, point_limit):
  match = Match([white_player_cls(Color.White), black_player_cls(Color.Black)], point_limit)
  match.start_game()
  while True:
    if match.tick():
      if match.winner is not None:
        game_count = len(match.games)
        print(f"Match ended! {match.winner.color} won with {match.points[match.winner.color]} points in {game_count} games with {match.turn_count} turns")
        for color, win_count in Counter(game.winner.color for game in match.games).items():
          print(f"{color} won {win_count} games ({win_count / game_count * 100}%)")
        return match
      else:
        game = match.current_game
        print(f"Game ended! {game.winner.color} won with {game.points} points after {game.turn_number} turns.")
        match.start_game()


if __name__ == "__main__":
  args = parser.parse_args()
  print(args.white)
  players = [globals()[f"{name}Player"] for name in (args.white, args.black)]
  match = _run_match(*players, args.match_points)
  sys.exit(0)
