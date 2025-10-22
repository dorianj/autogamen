"""Reinforcement learning training - migrated from run_reinforcement_learning.py"""
import cProfile
import glob
import io
import itertools
import os.path
import pstats
import sys
import time
from datetime import datetime

import torch

from autogamen.ai.mlp import MLPPlayer, Net
from autogamen.ai.simple import BozoPlayer, DeltaPlayer
from autogamen.game.game import Game
from autogamen.game.match import Match
from autogamen.game.types import Color


def _fmt_percent(p: float) -> str:
    return f"{p:.1%}"


def net_directory() -> str:
    """Get the directory for storing network files"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'nets')


def write_checkpoint(path: str, net: "Net", game_count: int) -> None:
    """Save training checkpoint"""
    torch.save({
        'model_state_dict': net.state_dict(),
        'eligibility': net.eligibility_traces,
        'game_count': game_count,
        'time': datetime.now().isoformat()
    }, path)


def load_checkpoint(path: str) -> tuple["Net", dict]:
    """Load checkpoint. Returns (Net, checkpoint_raw_info)"""
    checkpoint = torch.load(path)
    n = Net()
    n.eligibility_traces = checkpoint['eligibility']
    n.load_state_dict(checkpoint['model_state_dict'])
    return n, checkpoint


def run_game(white: "MLPPlayer", black: "MLPPlayer", net: "Net") -> None:
    """Run a single training game"""
    game = Game([white, black])
    game.start()

    for _turn_number in itertools.count():
        game.run_turn()
        if game.winner is not None:
            return


def run_exhib_match(net: "Net", cls: type) -> None:
    """Run exhibition match for evaluation"""
    match = Match([MLPPlayer(Color.White, net, False), cls(Color.Black)], 15)
    match.start_game()

    while True:
        if match.tick():
            if match.winner is not None:
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


def run_reinforcement_learning(
    games: int,
    checkpoint_interval: int,
    alpha: float,
    profile: bool,
    exhibition: bool,
) -> None:
    """Main entry point for reinforcement learning training"""
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    run_timestamp = datetime.now().isoformat()
    print(f"Running learning for {games} games, checkpointing every "
          f"{checkpoint_interval} games to timestamp {run_timestamp}")

    # Ensure nets directory exists
    net_dir = net_directory()
    os.makedirs(net_dir, exist_ok=True)

    # Try to find a checkpoint file to load
    checkpoint_paths = sorted(glob.glob(os.path.join(net_dir, "*.torch")))
    if len(checkpoint_paths):
        net, checkpoint = load_checkpoint(checkpoint_paths[-1])
        gen = checkpoint['game_count']
        print(f"Loaded checkpoint file with {checkpoint['game_count']} games from {checkpoint['time']}")
    else:
        print("Training net from scratch...")
        net = Net()
        gen = 0

    start_time = time.perf_counter()
    white, black = MLPPlayer(Color.White, net, True), MLPPlayer(Color.Black, net, True)
    for i in range(0, games + 1):
        if i > 0 and i % checkpoint_interval == 0:
            print(f"Checkpointing at {i}")
            checkpoint_path = os.path.join(net_dir, f"net-{run_timestamp}-{i+gen:07d}.torch")
            write_checkpoint(checkpoint_path, net, i)

            if exhibition:
                run_exhib_match(net, BozoPlayer)
                run_exhib_match(net, DeltaPlayer)

        run_game(white, black, net)

    print(f"Finished, {(time.perf_counter() - start_time) / games}s per game")

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats(0.2)
        print(s.getvalue())

    sys.exit(0)
