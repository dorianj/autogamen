import argparse
import itertools
import logging
import math
import os.path
import pickle
import random
import sys
from collections import Counter
from datetime import datetime
from multiprocessing import Pool

from autogamen.ai.mlp import MLPPlayer, Net
from autogamen.game.match import Match
from autogamen.game.types import Color

parser = argparse.ArgumentParser()
parser.add_argument("--generations", help="Number of generations to run", default=10, type=int)
parser.add_argument("--population", help="Number of players in each generation", default=10, type=int)
parser.add_argument("--parallelism", help="Number processes to use", default=1, type=int)
parser.add_argument("--mutation", help="Mutation rate", default=0.05, type=float)
parser.add_argument("--crossover", help="Percent of generation comprised of children", default=0.8, type=float)
parser.add_argument("--verbosity", help="Logging verbosity (debug/info/warning)", default="info")

args = parser.parse_args()

def _fmt_percent(p):
  return f"{p:.1%}"

def fitness_for_match(match):
  points = match.points[match.winner.color]
  return points - min(0.5, match.turn_count / 200)

def run_generation(generation, nets):
  print(f"Running generation {generation}...")
  player_count = len(nets)

  grouped_nets = (nets[i:i + 2] for i in range(0, len(nets), 2))
  match_players = list(
    (MLPPlayer(Color.White, white_net), MLPPlayer(Color.Black, black_net))
    for (white_net, black_net) in grouped_nets
  )

  with Pool(processes=args.parallelism) as pool:
    completed_matches = pool.map(run_match_args, match_players)

  # Order matches by their fitness
  matches_by_fitness = sorted(completed_matches, key=fitness_for_match, reverse=True)
  print(f"matches_by_fitness: {matches_by_fitness}")

  # Elite players are passed as-is
  elite_matches = matches_by_fitness[0:math.ceil(player_count * (1 - args.crossover / 2))]
  elite_nets = (match.winner.net for match in elite_matches)

  # Elite players are also included as mutated
  elite_mutated_nets = (net.mutate(args.mutation) for net in elite_nets)

  # To fill the remaining slots, breed players from the top half
  eligible_parents = [match.winner.net for match in matches_by_fitness[0:len(matches_by_fitness) // 2]]
  offspring_nets = []
  for _i in range(0, math.ceil(player_count * args.crossover)):
    [p1, p2] = random.choices(eligible_parents, k=2)
    offspring_nets.append(p1.breed(p2))

  next_generation = list(itertools.chain(elite_nets, elite_mutated_nets, offspring_nets))

  print(f"next_generation: {len(next_generation)}")
  # trim off any extras in case of off-by-one
  return next_generation[0:args.population]

def run_match_args(a):
  numeric_level = getattr(logging, args.verbosity.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {args.verbosity}')
  logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

  return run_match(*a)

def run_match(white_player, black_player):
  match = Match([white_player, black_player], 1)
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


def net_directory():
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nets')


def write_net_to_file(nets, path):
  with open(path, 'wb') as fp:
    pickle.dump([net.weights for net in nets], fp)


if __name__ == "__main__":
  # Housekeeping: log levels
  numeric_level = getattr(logging, args.verbosity.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {args.verbosity}')
  logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")

  if args.population % 2 == 1:
    raise Exception("--population must be even")

  run_timestamp = datetime.now().isoformat()
  print(f"Running evolution with {args.generations} generations and population "
        f"{args.population}, for a total of {args.generations * args.population // 2} games. "
        f" Writing results to timestamp {run_timestamp}")

  # Run the generations.
  nets = [Net.random_net() for n in range(0, args.population)]
  for gen in range(0, args.generations):
    nets = run_generation(gen, nets)
    write_net_to_file(nets, os.path.join(net_directory(), f"net-{run_timestamp}-{gen}.pickle"))

  sys.exit(0)
