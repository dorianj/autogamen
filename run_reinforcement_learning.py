from datetime import datetime
import argparse
import glob
import itertools
import logging
import os.path
import sys
import time

import torch

from autogamen.ai.mlp import MLPPlayer, Net
from autogamen.ai.simple import BozoPlayer, DeltaPlayer
from autogamen.game.game import Game
from autogamen.game.match import Match
from autogamen.game.types import Color

parser = argparse.ArgumentParser()
parser.add_argument("--games", help="Number of games to play", default=10, type=int)
parser.add_argument("--checkpoint", help="Save model after this many training games", default=1000, type=int)
parser.add_argument("--alpha", help="Learning rate", default=0.1, type=float)
parser.add_argument("--profile", help="Run Python profiler", default=False, type=bool)
parser.add_argument("--verbosity", help="Logging verbosity (debug/info/warning)", default="info")
args = parser.parse_args()


def _fmt_percent(p):
  return "{0:.1%}".format(p)


def net_directory():
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nets')


def write_checkpoint(path, net, game_count):
  torch.save({
    'model_state_dict': net.state_dict(),
    'eligibility': net.eligibility_traces,
    'game_count': game_count,
    'time': datetime.now().isoformat()
  }, path)


def load_checkpoint(path):
  """Returns (Net, checkpoint_raw_info)
  """
  checkpoint = torch.load(path)
  n = Net()
  n.eligibility_traces = checkpoint['eligibility']
  n.load_state_dict(checkpoint['model_state_dict'])
  return n, checkpoint


def run_game(white, black, net):
  game = Game([white, black])
  game.start()

  for turn_number in itertools.count():
    game.run_turn()
    if game.winner is not None:
      return

def run_exhib_match(net, cls):
  match = Match([MLPPlayer(Color.White, net, False), cls(Color.Black)], 15)
  match.start_game()

  while True:
    if match.tick():
      if match.winner is not None:
        game_count = len(match.games)
        print("{} exhibition vs {}, {} to {} in {} turns".format(
          "WON" if match.winner.color == Color.White else "Lost",
          cls.__name__,
          match.points[Color.White],
          match.points[Color.Black],
          match.turn_count,
        ))
        return
      else:
        match.start_game()


if __name__ == "__main__":
  if args.profile:
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

  run_timestamp = datetime.now().isoformat()
  print(f"Running learning for {args.games} games, checkpointing every "
        f"{args.checkpoint} games to timestamp {run_timestamp}")

  # Housekeeping: log levels
  numeric_level = getattr(logging, args.verbosity.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
  logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

  # Try to find a checkpoint file to load
  checkpoint_paths = sorted(glob.glob(os.path.join(net_directory(), "*.torch")))
  if len(checkpoint_paths):
    net, checkpoint = load_checkpoint(checkpoint_paths[-1])
    gen = checkpoint['game_count']
    print(f"Loaded checkpoint file with {checkpoint['game_count']} games from {checkpoint['time']}")
  else:
    print(f"Training net from scratch...")
    net = Net()
    gen = 0

  start_time = time.perf_counter()
  [white, black] = [MLPPlayer(Color.White, net, True), MLPPlayer(Color.Black, net, True)]
  for i in range(0, args.games + 1):
    if i > 0 and i % args.checkpoint == 0:
      print(f"Checkpointing at {i}")
      checkpoint_path = os.path.join(net_directory(), f"net-{run_timestamp}-{i+gen:07d}.torch")
      write_checkpoint(checkpoint_path, net, i)

      run_exhib_match(net, BozoPlayer)
      run_exhib_match(net, DeltaPlayer)

    run_game(white, black, net)

  print(f"Finished, {(time.perf_counter() - start_time) / args.games}s per game")

  if args.profile:
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
    ps.print_stats(0.2)
    print(s.getvalue())

  sys.exit(0)
