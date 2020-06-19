from collections import Counter
from datetime import datetime
import argparse
import itertools
import logging
import math
import os.path
import pickle
import random
import sys
import time

from autogamen.ai.mlp import MLPPlayer, Net
from autogamen.game.game import Game
from autogamen.game.match import Match
from autogamen.game.types import Color

parser = argparse.ArgumentParser()
parser.add_argument("--games", help="Number of games to play", default=10, type=int)
parser.add_argument("--checkpoint", help="Save model after this many training games", default=1000, type=int)
parser.add_argument("--alpha", help="Learning rate", default=0.1, type=float)
parser.add_argument("--verbosity", help="Logging verbosity (debug/info/warning)", default="info")

args = parser.parse_args()

def _fmt_percent(p):
  return "{0:.1%}".format(p)


def net_directory():
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nets')


def write_net_to_file(nets, path):
  with open(path, 'wb') as fp:
    pickle.dump([net.weights for net in nets], fp)


def run_game(white, black, net):
  game = Game([white, black])
  game.start()

  for turn_number in itertools.count():
    game.run_turn()
    if game.winner is not None:
      return


if __name__ == "__main__":
  run_timestamp = datetime.now().isoformat()
  print(f"Running learning for {args.games} games, checkpointing every "
        f"{args.checkpoint} games to timestamp {run_timestamp}")

  # Housekeeping: log levels
  numeric_level = getattr(logging, args.verbosity.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
  logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

  start_time = time.perf_counter()
  net = Net()
  [white, black] = [MLPPlayer(Color.White, net, True), MLPPlayer(Color.Black, net, True)]
  for i in range(0, args.games + 1):
    if i > 0 and i % args.checkpoint == 0:
      print(f"Checkpointing at {i}")
      #write_net_to_file(nets, os.path.join(net_directory(), f"net-{run_timestamp}-{gen}.pickle"))

    run_game(white, black, net)

  print(f"Finished, {(time.perf_counter() - start_time) / args.games}s per game")
  sys.exit(0)
