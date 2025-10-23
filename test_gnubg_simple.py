"""simple test of gnubg interface"""
from autogamen.game.board import Board
from autogamen.game.game_types import Color, Point
from autogamen.gnubg.interface import GnubgInterface


def main() -> None:
    # create starting board
    board = Board([
        Point(2, Color.White),
        Point(),
        Point(),
        Point(),
        Point(),
        Point(5, Color.Black),
        Point(),
        Point(3, Color.Black),
        Point(),
        Point(),
        Point(),
        Point(5, Color.White),
        Point(5, Color.Black),
        Point(),
        Point(),
        Point(),
        Point(3, Color.White),
        Point(),
        Point(5, Color.White),
        Point(),
        Point(),
        Point(),
        Point(),
        Point(2, Color.Black),
    ])

    print("starting gnubg...")
    gnubg = GnubgInterface()
    gnubg.start()

    print("getting hint for opening roll 3-1...")
    hints = gnubg.get_hint(board, Color.White, (3, 1))

    print(f"got {len(hints)} suggestions:")
    for i, hint in enumerate(hints[:3]):
        print(f"  {i+1}. {hint.moves} (equity: {hint.equity:.3f}, win: {hint.win_prob:.3f})")

    print("stopping gnubg...")
    gnubg.stop()
    print("done!")


if __name__ == "__main__":
    main()
