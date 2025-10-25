#!/usr/bin/env python3
"""flexible battle framework for testing arbitrary player combinations."""
import argparse
import glob
import os
import sys

from autogamen.ai.mlp import MLPPlayer
from autogamen.ai.players import (
    BozoPlayer,
    DeltaPlayer,
    GnubgPlayer,
    Player,
    PypibgPlayer,
    RunningPlayer,
)
from autogamen.game.game_types import Color
from autogamen.game.match import Match
from autogamen.game.reinforcement_learning import load_checkpoint, net_directory


def parse_player_spec(spec: str, color: Color) -> Player:
    """parse player specification string into Player instance.

    supported formats:
      random              → BozoPlayer
      running             → RunningPlayer
      delta               → DeltaPlayer
      gnubg:N             → GnubgPlayer(plies=N)
      pypibg              → PypibgPlayer (fast neural net)
      fast-gnubg          → PypibgPlayer (deprecated, use pypibg)
      mlp:path.torch      → MLPPlayer loaded from checkpoint
      mlp:latest          → MLPPlayer from latest checkpoint
    """
    spec = spec.strip().lower()

    if spec == "random":
        return BozoPlayer(color)
    elif spec == "running":
        return RunningPlayer(color)
    elif spec == "delta":
        return DeltaPlayer(color)
    elif spec.startswith("gnubg:"):
        try:
            plies = int(spec.split(":")[1])
            return GnubgPlayer(color, plies=plies)
        except (IndexError, ValueError) as e:
            raise ValueError(f"invalid gnubg spec '{spec}' (expected gnubg:N where N is ply count)") from e
    elif spec == "pypibg":
        return PypibgPlayer(color)
    elif spec == "fast-gnubg":
        # deprecated alias for backward compatibility
        return PypibgPlayer(color)
    elif spec.startswith("mlp:"):
        path = spec.split(":", 1)[1]

        if path == "latest":
            # find latest checkpoint
            checkpoint_paths = sorted(glob.glob(os.path.join(net_directory(), "*.torch")))
            if not checkpoint_paths:
                raise RuntimeError("no checkpoints found in out/train/")
            path = checkpoint_paths[-1]
            print(f"ℹ loading latest checkpoint: {os.path.basename(path)}")
        else:
            # expand relative paths
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)

        net, checkpoint = load_checkpoint(path)
        print(f"  trained on {checkpoint['game_count']:,} games ({checkpoint['time']})")
        return MLPPlayer(color, net, learning=False)
    else:
        raise ValueError(
            f"unknown player spec: {spec}\n"
            f"supported: random, running, delta, gnubg:N, pypibg, mlp:path, mlp:latest"
        )


def run_match(white: Player, black: Player, point_goal: int, verbose: bool = True) -> tuple[Color, int, int]:
    """run a single match. returns (winner_color, white_points, black_points)"""
    match = Match([white, black], point_goal)

    while match.winner is None:
        game_over = match.tick()

        if verbose and game_over and match.current_game is None:
            prev_game = match.games[-1]
            print(
                f"  game {len(match.games)}: {prev_game.winner.color.name.lower()} wins "
                f"(+{prev_game.points} pts, {prev_game.turn_number} turns) → "
                f"score: {match.points[Color.White]}-{match.points[Color.Black]}"
            )

    return match.winner.color, match.points[Color.White], match.points[Color.Black]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run battles between arbitrary backgammon players",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --white gnubg:0 --black gnubg:2         # easy vs hard gnubg
  %(prog)s --white mlp:latest --black gnubg:2      # trained mlp vs gnubg
  %(prog)s --white pypibg --black random           # fast neural net vs random
  %(prog)s --white running --black delta -n 10     # heuristics comparison
        """,
    )
    parser.add_argument("--white", required=True, help="white player spec (see examples)")
    parser.add_argument("--black", required=True, help="black player spec (see examples)")
    parser.add_argument("-n", "--num-matches", type=int, default=1, help="number of matches to play")
    parser.add_argument("-p", "--points", type=int, default=7, help="match length in points")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress per-game output")
    args = parser.parse_args()

    # create players
    print("§ setting up players...")
    try:
        white = parse_player_spec(args.white, Color.White)
        black = parse_player_spec(args.black, Color.Black)
    except (ValueError, RuntimeError) as e:
        print(f"✘ error: {e}", file=sys.stderr)
        sys.exit(1)

    # display match setup
    print(f"\n→ running {args.num_matches} match{'es' if args.num_matches != 1 else ''} ({args.points}-point)")
    print(f"  white: {args.white}")
    print(f"  black: {args.black}")
    print()

    # track results across matches
    white_wins = 0
    black_wins = 0
    total_white_pts = 0
    total_black_pts = 0

    for i in range(args.num_matches):
        if args.num_matches > 1:
            print(f"⏵ match {i+1}/{args.num_matches}")

        winner, white_pts, black_pts = run_match(white, black, args.points, verbose=not args.quiet)

        total_white_pts += white_pts
        total_black_pts += black_pts

        if winner == Color.White:
            white_wins += 1
        else:
            black_wins += 1

        if args.num_matches > 1:
            print(f"  result: {winner.name.lower()} wins {white_pts}-{black_pts}")
            print()

    # final results
    print("✔ battle complete")
    if args.num_matches > 1:
        total_matches = white_wins + black_wins
        print(f"  white: {white_wins}/{total_matches} matches ({white_wins/total_matches:.1%})")
        print(f"  black: {black_wins}/{total_matches} matches ({black_wins/total_matches:.1%})")
        print(f"  avg points: white {total_white_pts/args.num_matches:.1f}, black {total_black_pts/args.num_matches:.1f}")
    else:
        print(f"  winner: {winner.name.lower()}")
        print(f"  final score: {white_pts}-{black_pts}")


if __name__ == "__main__":
    main()
