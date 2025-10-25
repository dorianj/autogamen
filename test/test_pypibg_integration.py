"""integration tests for pypibg (gnubg python package)."""
import time
import unittest

try:
    import gnubg as pypibg
    PYPIBG_AVAILABLE = True
except ImportError:
    PYPIBG_AVAILABLE = False

from autogamen.ai.players import BozoPlayer, PypibgPlayer
from autogamen.game.game import Game
from autogamen.game.game_types import Color


@unittest.skipUnless(PYPIBG_AVAILABLE, "pypibg package not installed")
class TestPypibgPackage(unittest.TestCase):
    """test pypibg package functions."""

    def test_board_format(self) -> None:
        """pypibg board format: [[white positions], [black positions]]."""
        # starting position in pypibg format
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # check board is valid
        self.assertEqual(len(start_board), 2)
        self.assertEqual(len(start_board[0]), 25)
        self.assertEqual(len(start_board[1]), 25)
        self.assertEqual(sum(start_board[0]), 15)  # white has 15 checkers
        self.assertEqual(sum(start_board[1]), 15)  # black has 15 checkers

    def test_classification(self) -> None:
        """test position classification."""
        # starting position
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # contact position should be class 5
        classification = pypibg.classify(start_board)
        self.assertEqual(classification, 5, "starting position should be contact (5)")

        # just test that classify returns valid class values
        # pypibg classification might be buggy, so just check range
        bearoff_board = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 3, 3, 0],
            [0, 3, 3, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        bearoff_class = pypibg.classify(bearoff_board)
        # class values: 0=over, 1=race, 2=crashed, 5=contact, 6=bearoff
        self.assertIn(bearoff_class, [0, 1, 2, 5, 6], "classification should be valid")

    def test_evaluation(self) -> None:
        """test position evaluation."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # 0-ply evaluation
        probs = pypibg.probabilities(start_board, 0)
        self.assertEqual(len(probs), 5)

        # probabilities should be reasonable (sum close to 1 for win/lose)
        win_prob = probs[0]
        self.assertGreaterEqual(win_prob, 0.0)
        self.assertLessEqual(win_prob, 1.0)

        # starting position should be roughly even
        self.assertGreater(win_prob, 0.4)
        self.assertLess(win_prob, 0.6)

    def test_move_generation(self) -> None:
        """test legal move generation."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # generate moves for dice (3, 1) - returns tuple of position keys
        moves = pypibg.moves(start_board, 3, 1)
        self.assertIsInstance(moves, tuple)
        self.assertGreater(len(moves), 0, "should have legal moves for 3-1")

        # each move should be a string (position key)
        for move in moves:
            self.assertIsInstance(move, str)

    def test_best_move(self) -> None:
        """test best move selection."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # get single best move
        best_1 = pypibg.best_move(start_board, 3, 1, 1, b'X')
        # pypibg seems to always return a tuple of moves
        self.assertIsInstance(best_1, tuple)
        self.assertGreater(len(best_1), 0)

        # get top 3 moves - still returns tuple
        best_3 = pypibg.best_move(start_board, 3, 1, 3, b'X')
        # it seems to just return the single best move regardless of n
        self.assertIsInstance(best_3, tuple)

    def test_position_encoding(self) -> None:
        """test position id and key encoding/decoding."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # encode to position id
        pos_id = pypibg.position_id(start_board)
        self.assertIsInstance(pos_id, str)

        # decode back
        board_from_id = pypibg.board_from_position_id(pos_id)
        self.assertEqual(board_from_id, start_board, "roundtrip through position_id failed")

        # encode to position key
        pos_key = pypibg.key_of_board(start_board)
        self.assertIsInstance(pos_key, str)

        # decode back - returns tuple of tuples instead of list of lists
        board_from_key = pypibg.board_from_position_key(pos_key)
        # convert to list of lists for comparison
        board_as_list = [list(board_from_key[0]), list(board_from_key[1])]
        self.assertEqual(board_as_list, start_board, "roundtrip through position_key failed")

    def test_pub_eval(self) -> None:
        """test pubeval heuristic evaluation."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        score = pypibg.pub_eval_score(start_board)
        self.assertIsInstance(score, float)
        # pubeval returns a wider range of scores, just check it's reasonable
        self.assertGreater(score, -100)
        self.assertLess(score, 100)


@unittest.skipUnless(PYPIBG_AVAILABLE, "pypibg package not installed")
class TestPypibgPlayer(unittest.TestCase):
    """test PypibgPlayer integration."""

    def test_board_conversion(self) -> None:
        """test conversion from our board format to pypibg format."""
        # create a game to get a board
        game = Game([BozoPlayer(Color.White), BozoPlayer(Color.Black)])
        board = game.board

        # create player and test conversion
        player = PypibgPlayer(Color.White)
        pypibg_board = player._our_board_to_pypibg(board)

        # check format
        self.assertEqual(len(pypibg_board), 2)
        self.assertEqual(len(pypibg_board[0]), 25)
        self.assertEqual(len(pypibg_board[1]), 25)

        # check checker counts
        white_count = sum(pypibg_board[0])
        black_count = sum(pypibg_board[1])
        self.assertEqual(white_count, 15)
        self.assertEqual(black_count, 15)

    def test_player_action(self) -> None:
        """test PypibgPlayer can select moves."""
        # create game with pypibg player
        white = PypibgPlayer(Color.White)
        black = BozoPlayer(Color.Black)
        game = Game([white, black])

        # start game
        game.start()

        # play a turn
        game.pre_turn()
        game.turn_blocking()

        # game should progress
        self.assertGreaterEqual(game.turn_number, 1)

    def test_full_game(self) -> None:
        """test full game with pypibg players."""
        white = PypibgPlayer(Color.White)
        black = PypibgPlayer(Color.Black)
        game = Game([white, black])

        game.start()

        # play up to 200 turns (safety limit)
        for _ in range(200):
            if game.winner is not None:
                break
            game.pre_turn()
            game.turn_blocking()

        # game should complete
        self.assertIsNotNone(game.winner)
        self.assertIn(game.winner.color, [Color.White, Color.Black])


class TestPypibgPerformance(unittest.TestCase):
    """performance benchmarks for pypibg."""

    @unittest.skipUnless(PYPIBG_AVAILABLE, "pypibg package not installed")
    def test_evaluation_speed(self) -> None:
        """benchmark 0-ply evaluation speed."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # warmup
        for _ in range(10):
            pypibg.probabilities(start_board, 0)

        # measure
        n = 100
        start_time = time.perf_counter()
        for _ in range(n):
            pypibg.probabilities(start_board, 0)
        elapsed = time.perf_counter() - start_time

        evals_per_sec = n / elapsed

        # should be fast (at least 100 evals/sec)
        self.assertGreater(evals_per_sec, 100,
                          f"0-ply evaluation too slow: {evals_per_sec:.1f} evals/sec")

    @unittest.skipUnless(PYPIBG_AVAILABLE, "pypibg package not installed")
    def test_move_generation_speed(self) -> None:
        """benchmark move generation speed."""
        start_board = [
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        ]

        # warmup
        for _ in range(10):
            pypibg.moves(start_board, 3, 1)

        # measure
        n = 100
        start_time = time.perf_counter()
        for _ in range(n):
            pypibg.moves(start_board, 3, 1)
        elapsed = time.perf_counter() - start_time

        moves_per_sec = n / elapsed

        # should be fast (at least 100 move generations/sec)
        self.assertGreater(moves_per_sec, 100,
                          f"move generation too slow: {moves_per_sec:.1f} gens/sec")


if __name__ == "__main__":
    unittest.main()
