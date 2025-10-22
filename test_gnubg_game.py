"""test playing a game with gnubg player."""

from autogamen.ai.players import BozoPlayer, GnubgPlayer
from autogamen.game.game import Game
from autogamen.game.game_types import Color


def play_test_game():
    """play a short test game with gnubg."""
    print("starting test game: gnubg (white) vs bozo (black)")

    white_player = GnubgPlayer(Color.White)
    black_player = BozoPlayer(Color.Black)

    game = Game([white_player, black_player])
    game.start()

    print(f"ℹ starting player: {game.active_color.name} (rolled {game.active_dice.roll})")

    max_turns = 20  # limit to 20 turns to keep test short
    turn_count = 0

    while not game.winner and turn_count < max_turns:
        game.pre_turn()  # advance to next player and roll dice
        turn_count += 1

        current_player = game.active_player()
        player_name = "gnubg" if game.active_color == Color.White else "bozo"

        print(f"\n→ turn {turn_count}: {player_name} ({game.active_color.name}) rolls {game.active_dice.roll}")

        possible_moves = game.board.possible_moves(game.active_color, game.active_dice)

        if not possible_moves:
            print("  no moves available, passing")
            game.turn_blocking()
            continue

        action = current_player.action(possible_moves)

        if action[0].name == "Move":
            moves = action[1]
            print(f"  selected: {' '.join(str(m) for m in moves)}")

        game.turn_blocking()

    if game.winner:
        winner_name = "gnubg" if game.winner.color == Color.White else "bozo"
        print(f"\n✱ {winner_name} ({game.winner.color.name}) wins with {game.points} points!")
    else:
        print(f"\n⏹ game ended after {max_turns} turns (no winner yet)")

    # cleanup
    white_player.__del__()

    print("\n✔ test completed successfully")


if __name__ == "__main__":
    play_test_game()
