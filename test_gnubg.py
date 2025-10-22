"""test script for gnubg integration."""

from autogamen.game.board import Board
from autogamen.game.game_types import Color, FrozenPoint
from autogamen.gnubg.interface import GnubgInterface


def test_initial_position():
    """test getting hints for the initial backgammon position."""
    print("testing gnubg interface with initial position...")

    # create initial backgammon position
    # standard starting position:
    # point 1: 2 white, point 6: 5 black, point 8: 3 black
    # point 12: 5 white, point 13: 5 black, point 17: 3 white
    # point 19: 5 white, point 24: 2 black
    points = [FrozenPoint(0, None)] * 24

    # white pieces (moving from 24 to 1)
    points[0] = FrozenPoint(2, Color.White)   # point 1
    points[11] = FrozenPoint(5, Color.White)  # point 12
    points[16] = FrozenPoint(3, Color.White)  # point 17
    points[18] = FrozenPoint(5, Color.White)  # point 19

    # black pieces (moving from 1 to 24)
    points[5] = FrozenPoint(5, Color.Black)   # point 6
    points[7] = FrozenPoint(3, Color.Black)   # point 8
    points[12] = FrozenPoint(5, Color.Black)  # point 13
    points[23] = FrozenPoint(2, Color.Black)  # point 24

    board = Board(points)

    # test for white player with dice roll 2,4
    dice = (2, 4)

    with GnubgInterface() as gnubg:
        hints = gnubg.get_hint(board, Color.White, dice)

        print(f"\ngnubg returned {len(hints)} move suggestions for dice {dice}:")
        for i, hint in enumerate(hints[:5], 1):  # show top 5
            print(f"{i}. {hint.moves:30s} eq: {hint.equity:+.3f}  "
                  f"win: {hint.win_prob:.3f}  lose: {hint.lose_prob:.3f}")

    print("\n✔ gnubg integration test passed!")


if __name__ == "__main__":
    test_initial_position()
