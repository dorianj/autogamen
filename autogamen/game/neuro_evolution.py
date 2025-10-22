"""Neuro-evolution training - migrated from run_neuro_evolution.py"""
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
from autogamen.game.game_types import Color
from autogamen.game.match import Match


def _fmt_percent(p: float) -> str:
    return f"{p:.1%}"


def fitness_for_match(match: "Match") -> float:
    """Calculate fitness score for a match"""
    points = match.points[match.winner.color]
    return points - min(0.5, match.turn_count / 200)


def run_match_args(args: tuple) -> "Match":
    """Wrapper to configure logging in subprocess"""
    verbosity, white_player, black_player = args
    numeric_level = getattr(logging, verbosity.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {verbosity}')
    logging.basicConfig(level=numeric_level, format="%(asctime)s: %(message)s")
    return run_match(white_player, black_player)


def run_match(white_player: "MLPPlayer", black_player: "MLPPlayer") -> "Match":
    """Run a single match to completion"""
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


def run_generation(generation: int, nets: list["Net"], parallelism: int, mutation: float, crossover: float, population: int) -> list["Net"]:
    """Run a single generation of evolution"""
    print(f"Running generation {generation}...")
    player_count = len(nets)

    grouped_nets = (nets[i:i + 2] for i in range(0, len(nets), 2))
    match_players = list(
        (MLPPlayer(Color.White, white_net), MLPPlayer(Color.Black, black_net))
        for (white_net, black_net) in grouped_nets
    )

    verbosity = logging.getLevelName(logging.getLogger().getEffectiveLevel()).lower()

    with Pool(processes=parallelism) as pool:
        args_list = [(verbosity, white, black) for white, black in match_players]
        completed_matches = pool.map(run_match_args, args_list)

    # Order matches by their fitness
    matches_by_fitness = sorted(completed_matches, key=fitness_for_match, reverse=True)
    print(f"matches_by_fitness: {matches_by_fitness}")

    # Elite players are passed as-is
    elite_matches = matches_by_fitness[0:math.ceil(player_count * (1 - crossover / 2))]
    elite_nets = (match.winner.net for match in elite_matches)

    # Elite players are also included as mutated
    elite_mutated_nets = (net.mutate(mutation) for net in elite_nets)

    # To fill the remaining slots, breed players from the top half
    eligible_parents = [match.winner.net for match in matches_by_fitness[0:len(matches_by_fitness) // 2]]
    offspring_nets = []
    for _i in range(0, math.ceil(player_count * crossover)):
        [p1, p2] = random.choices(eligible_parents, k=2)
        offspring_nets.append(p1.breed(p2))

    next_generation = list(itertools.chain(elite_nets, elite_mutated_nets, offspring_nets))

    print(f"next_generation: {len(next_generation)}")
    # trim off any extras in case of off-by-one
    return next_generation[0:population]


def net_directory() -> str:
    """Get the directory for storing network files"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'nets')


def write_net_to_file(nets: list["Net"], path: str) -> None:
    """Save networks to file"""
    with open(path, 'wb') as fp:
        pickle.dump([net.weights for net in nets], fp)


def run_neuro_evolution(
    generations: int,
    population: int,
    parallelism: int,
    mutation: float,
    crossover: float,
) -> None:
    """Main entry point for neuro-evolution training"""
    if population % 2 == 1:
        raise Exception("--population must be even")

    run_timestamp = datetime.now().isoformat()
    print(f"Running evolution with {generations} generations and population "
          f"{population}, for a total of {generations * population // 2} games. "
          f" Writing results to timestamp {run_timestamp}")

    # Ensure nets directory exists
    net_dir = net_directory()
    os.makedirs(net_dir, exist_ok=True)

    # Run the generations
    nets = [Net.random_net() for _ in range(0, population)]
    for gen in range(0, generations):
        nets = run_generation(gen, nets, parallelism, mutation, crossover, population)
        write_net_to_file(nets, os.path.join(net_dir, f"net-{run_timestamp}-{gen}.pickle"))

    sys.exit(0)
