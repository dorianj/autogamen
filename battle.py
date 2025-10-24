"""run a battle between RL-trained neural net and gnubg."""
import argparse
import glob
import os

from autogamen.ai.mlp import MLPPlayer
from autogamen.ai.players import GnubgPlayer
from autogamen.game.game_types import Color
from autogamen.game.match import Match
from autogamen.game.reinforcement_learning import load_checkpoint, net_directory


def run_single_match(mlp: MLPPlayer, gnubg: GnubgPlayer, point_goal: int, verbose: bool = True) -> tuple[Color, int, int]:
    """run a single match. returns (winner_color, white_points, black_points)"""
    match = Match([mlp, gnubg], point_goal)

    while match.winner is None:
        game_over = match.tick()

        # show progress when game completes
        if verbose and game_over and match.current_game is None:
            prev_game = match.games[-1]
            print(f"  game {len(match.games)}: {prev_game.winner.color.name.lower()} wins "
                  f"(+{prev_game.points} pts, {prev_game.turn_number} turns) → score: {match.points[Color.White]}-{match.points[Color.Black]}")

    return match.winner.color, match.points[Color.White], match.points[Color.Black]


def main() -> None:
    parser = argparse.ArgumentParser(description="run battles between RL-trained net and gnubg")
    parser.add_argument("-n", "--num-matches", type=int, default=1, help="number of matches to play")
    parser.add_argument("-p", "--points", type=int, default=7, help="match length in points")
    parser.add_argument("--plies", type=int, default=7, help="gnubg evaluation plies (default: 7 = world class)")
    args = parser.parse_args()
    # find latest checkpoint
    checkpoint_paths = sorted(glob.glob(os.path.join(net_directory(), "*.torch")))
    if not checkpoint_paths:
        raise RuntimeError("no checkpoints found in out/models/")

    latest = checkpoint_paths[-1]
    print(f"loading checkpoint: {os.path.basename(latest)}")

    net, checkpoint = load_checkpoint(latest)
    print(f"  trained on {checkpoint['game_count']:,} games")
    print(f"  timestamp: {checkpoint['time']}")

    # set up players
    print("\n⏵ setting up players...")
    mlp = MLPPlayer(Color.White, net, learning=False)
    gnubg = GnubgPlayer(Color.Black, plies=args.plies)

    print(f"\n§ running {args.num_matches} match{'es' if args.num_matches > 1 else ''} ({args.points}-point)")
    print(f"  white: MLPPlayer (RL-trained, {checkpoint['game_count']:,} games)")
    print(f"  black: GnubgPlayer (gnu backgammon {args.plies}-ply)")
    print()

    # track results across matches
    mlp_wins = 0
    gnubg_wins = 0

    for i in range(args.num_matches):
        if args.num_matches > 1:
            print(f"→ match {i+1}/{args.num_matches}")

        winner, white_pts, black_pts = run_single_match(mlp, gnubg, args.points, verbose=True)

        if winner == Color.White:
            mlp_wins += 1
        else:
            gnubg_wins += 1

        if args.num_matches > 1:
            print(f"  result: {winner.name.lower()} wins {white_pts}-{black_pts}")
            print()

    # final results
    print("✔ battle complete")
    if args.num_matches > 1:
        print(f"  mlp: {mlp_wins} wins ({mlp_wins/(mlp_wins+gnubg_wins):.1%})")
        print(f"  gnubg: {gnubg_wins} wins ({gnubg_wins/(mlp_wins+gnubg_wins):.1%})")


if __name__ == "__main__":
    main()
