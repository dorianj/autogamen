import argparse
import logging

from autogamen.ai.mlp import MLPPlayer
from autogamen.ai.simple import BozoPlayer, DeltaPlayer
from autogamen.ui.ui_player import HumanPlayer
from autogamen.game.match import Match
from autogamen.game.types import Color
from autogamen.ui.match_view import MatchView

parser = argparse.ArgumentParser()
parser.add_argument("opponent", default="Bozo", help="Class name of opponent")
parser.add_argument("--points", help="Point limit for the match to play", default=3, type=int)
parser.add_argument("--verbosity", help="Logging verbosity (debug/info/warning)", default="info")
args = parser.parse_args()

if __name__ == "__main__":
  # Housekeeping: log levels
  numeric_level = getattr(logging, args.verbosity.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
  logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

human_player = HumanPlayer(Color.White)
match = Match([human_player, DeltaPlayer(Color.Black)], 50)
match_view = MatchView(match)
match_view.create_window()
human_player.attach(match_view)
match_view.run()
