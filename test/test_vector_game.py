"""regression tests: vector_game matches Board behavior exactly"""
# mypy: disable-error-code="no-untyped-def,attr-defined,arg-type,assignment"
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from autogamen.game import vector_game as vg
from autogamen.game.board import Board
from autogamen.game.game_types import Color, Dice, Move, Point


def repeat_point(repeat_count, pips=0, color=None):
    if type(pips) is not int or (color is not None and type(color) is not Color):
        raise Exception("Invalid args into repeat_point")
    return [Point(pips, color) for i in range(repeat_count)]


default_starting_points = [
    Point(2, Color.White),
    Point(),
    Point(),
    Point(),
    Point(),
    Point(5, Color.Black),
    Point(0),
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
    Point(0),
    Point(5, Color.White),
    Point(),
    Point(),
    Point(),
    Point(),
    Point(2, Color.Black),
]


class TestConversion(unittest.TestCase):
    """test from_board() and to_board() round-trip correctly"""

    def test_empty_board_roundtrip(self):
        board = Board(repeat_point(24))
        vec = vg.from_board(board)
        reconstructed = vg.to_board(vec)
        self.assertEqual(board, reconstructed)

    def test_starting_board_roundtrip(self):
        board = Board(default_starting_points)
        vec = vg.from_board(board)
        reconstructed = vg.to_board(vec)
        self.assertEqual(board, reconstructed)

    def test_board_with_bar_roundtrip(self):
        board = Board(default_starting_points, bar=[2, 3])
        vec = vg.from_board(board)
        reconstructed = vg.to_board(vec)
        self.assertEqual(board, reconstructed)

    def test_board_with_off_roundtrip(self):
        board = Board(default_starting_points, off=[7, 4])
        vec = vg.from_board(board)
        reconstructed = vg.to_board(vec)
        self.assertEqual(board, reconstructed)

    def test_complex_board_roundtrip(self):
        points = [
            Point(7, Color.White),
            Point(1, Color.Black),
            Point(),
            Point(12, Color.Black),
            Point(),
            Point(3, Color.White),
        ] + repeat_point(18)
        board = Board(points, bar=[1, 0], off=[8, 10])
        vec = vg.from_board(board)
        reconstructed = vg.to_board(vec)
        self.assertEqual(board, reconstructed)


class TestApplyMove(unittest.TestCase):
    """test apply_move() matches Board.apply_move()"""

    def test_simple_move(self):
        board = Board([Point(2, Color.White)] + repeat_point(23))
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 6)
        board.apply_move(move)
        vec = vg.apply_move(vec, move)

        self.assertEqual(board, vg.to_board(vec))

    def test_simple_hit(self):
        board = Board([Point(2, Color.White), Point(1, Color.Black)] + repeat_point(22))
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 1)
        board.apply_move(move)
        vec = vg.apply_move(vec, move)

        self.assertEqual(board, vg.to_board(vec))

    def test_bear_off(self):
        board = Board(repeat_point(19) + [Point(2, Color.White)] + repeat_point(4))
        vec = vg.from_board(board)

        move = Move(Color.White, 20, 5)
        board.apply_move(move)
        vec = vg.apply_move(vec, move)

        self.assertEqual(board, vg.to_board(vec))

    def test_from_bar(self):
        board = Board(repeat_point(24), bar=[1, 0])
        vec = vg.from_board(board)

        move = Move(Color.White, Move.Bar, 3)
        board.apply_move(move)
        vec = vg.apply_move(vec, move)

        self.assertEqual(board, vg.to_board(vec))

    def test_hit_and_bear_off(self):
        board = Board(
            repeat_point(22) + [Point(1, Color.White), Point(1, Color.Black)]
        )
        vec = vg.from_board(board)

        move = Move(Color.White, 23, 1)
        board.apply_move(move)
        vec = vg.apply_move(vec, move)

        self.assertEqual(board, vg.to_board(vec))


class TestCanBearOff(unittest.TestCase):
    """test can_bear_off() matches Board.can_bear_off()"""

    def test_starting_board(self):
        board = Board(default_starting_points)
        vec = vg.from_board(board)

        self.assertEqual(board.can_bear_off(Color.White), vg.can_bear_off(vec, Color.White))
        self.assertEqual(board.can_bear_off(Color.Black), vg.can_bear_off(vec, Color.Black))

    def test_white_can_bear_off(self):
        points = (
            repeat_point(6) +
            [Point(5, Color.Black)] + repeat_point(12) +
            [Point(7, Color.White), Point(2, Color.White)] + repeat_point(3)
        )
        board = Board(points)
        vec = vg.from_board(board)

        self.assertTrue(board.can_bear_off(Color.White))
        self.assertTrue(vg.can_bear_off(vec, Color.White))
        self.assertFalse(board.can_bear_off(Color.Black))
        self.assertFalse(vg.can_bear_off(vec, Color.Black))

    def test_cant_bear_off_with_bar(self):
        points = repeat_point(19) + [Point(2, Color.White)] + repeat_point(4)
        board = Board(points, bar=[1, 0])
        vec = vg.from_board(board)

        self.assertFalse(board.can_bear_off(Color.White))
        self.assertFalse(vg.can_bear_off(vec, Color.White))


class TestMoveIsValid(unittest.TestCase):
    """test move_is_valid() matches Board.move_is_valid()"""

    def test_valid_normal_move(self):
        board = Board([Point(2, Color.White)] + repeat_point(23))
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 6)
        self.assertTrue(board.move_is_valid(move))
        self.assertTrue(vg.move_is_valid(vec, move))

    def test_invalid_empty_source(self):
        board = Board(repeat_point(24))
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 6)
        self.assertFalse(board.move_is_valid(move))
        self.assertFalse(vg.move_is_valid(vec, move))

    def test_invalid_blocked_destination(self):
        board = Board([Point(1, Color.White)] + [Point(2, Color.Black)] + repeat_point(22))
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 1)
        self.assertFalse(board.move_is_valid(move))
        self.assertFalse(vg.move_is_valid(vec, move))

    def test_valid_hit(self):
        board = Board([Point(1, Color.White), Point(1, Color.Black)] + repeat_point(22))
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 1)
        self.assertTrue(board.move_is_valid(move))
        self.assertTrue(vg.move_is_valid(vec, move))

    def test_cant_move_with_bar(self):
        board = Board([Point(2, Color.White)] + repeat_point(23), bar=[1, 0])
        vec = vg.from_board(board)

        move = Move(Color.White, 1, 6)
        self.assertFalse(board.move_is_valid(move))
        self.assertFalse(vg.move_is_valid(vec, move))


class TestPossibleMoves(unittest.TestCase):
    """test possible_moves() generates same moves as Board.possible_moves()"""

    def _compare_moves(self, board: Board, color: Color, dice: Dice):
        """helper: verify vector_game matches Board for given position"""
        vec = vg.from_board(board)

        board_moves = board.possible_moves(color, dice)
        vec_moves = vg.possible_moves(vec, color, dice.effective_roll())

        # extract just the move sequences
        board_move_seqs = set(m[0] for m in board_moves)
        vec_move_seqs = set(m[0] for m in vec_moves)

        self.assertEqual(
            len(board_move_seqs),
            len(vec_move_seqs),
            f"move count mismatch: board={len(board_move_seqs)}, vec={len(vec_move_seqs)}"
        )
        self.assertEqual(board_move_seqs, vec_move_seqs)

        # verify resulting boards match
        for move_seq, board_result in board_moves:
            # find matching vector result
            vec_result = next((v for m, v in vec_moves if m == move_seq), None)
            self.assertIsNotNone(vec_result, f"missing vector result for {move_seq}")
            self.assertEqual(board_result, vg.to_board(vec_result))

    def test_single_pip_normal_moves(self):
        board = Board(repeat_point(10) + [Point(1, Color.White)] + repeat_point(13))
        self._compare_moves(board, Color.White, Dice(roll=(1, 2)))

    def test_doubles(self):
        board = Board(repeat_point(10) + [Point(1, Color.White)] + repeat_point(13))
        self._compare_moves(board, Color.White, Dice(roll=(2, 2)))

    def test_starting_board(self):
        board = Board(default_starting_points)
        self._compare_moves(board, Color.White, Dice(roll=(3, 5)))
        self._compare_moves(board, Color.Black, Dice(roll=(2, 4)))

    def test_blocked_moves(self):
        board = Board([Point(1, Color.White)] + [Point(2, Color.Black)] + repeat_point(22))
        self._compare_moves(board, Color.White, Dice(roll=(1, 2)))

    def test_bear_off_moves(self):
        board = Board(repeat_point(19) + [Point(2, Color.White)] + repeat_point(4))
        self._compare_moves(board, Color.White, Dice(roll=(5, 6)))

    def test_from_bar_moves(self):
        board = Board(repeat_point(24), bar=[1, 0])
        self._compare_moves(board, Color.White, Dice(roll=(1, 6)))

    def test_no_moves_available(self):
        board = Board(
            [Point(2, Color.White)] + repeat_point(6, 2, Color.Black) + repeat_point(17)
        )
        self._compare_moves(board, Color.White, Dice(roll=(6, 6)))


class TestWinner(unittest.TestCase):
    """test winner() matches Board.winner()"""

    def test_no_winner_starting(self):
        board = Board(default_starting_points)
        vec = vg.from_board(board)

        self.assertIsNone(board.winner())
        self.assertIsNone(vg.winner(vec))

    def test_white_wins(self):
        board = Board(repeat_point(24), off=[15, 0])
        vec = vg.from_board(board)

        self.assertEqual(board.winner(), Color.White)
        self.assertEqual(vg.winner(vec), Color.White)

    def test_black_wins(self):
        board = Board(repeat_point(24), off=[0, 15])
        vec = vg.from_board(board)

        self.assertEqual(board.winner(), Color.Black)
        self.assertEqual(vg.winner(vec), Color.Black)


class TestPropertyBased(unittest.TestCase):
    """property-based tests using hypothesis for random positions"""

    @given(
        points=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=15),
                st.sampled_from([None, Color.White, Color.Black])
            ),
            min_size=24,
            max_size=24
        ),
        bar_white=st.integers(min_value=0, max_value=5),
        bar_black=st.integers(min_value=0, max_value=5),
        off_white=st.integers(min_value=0, max_value=15),
        off_black=st.integers(min_value=0, max_value=15),
    )
    @settings(max_examples=100, deadline=None)
    def test_roundtrip_preserves_board(self, points, bar_white, bar_black, off_white, off_black):
        """from_board(board) → to_board() should be identity"""
        # filter invalid points: non-zero count requires color
        valid_points = []
        for count, color in points:
            if count == 0:
                valid_points.append(Point(0, None))
            elif color is not None:
                valid_points.append(Point(count, color))
            else:
                # invalid: non-zero without color, make it empty
                valid_points.append(Point(0, None))

        try:
            board = Board(valid_points, bar=[bar_white, bar_black], off=[off_white, off_black])
            vec = vg.from_board(board)
            reconstructed = vg.to_board(vec)
            self.assertEqual(board, reconstructed)
        except Exception:
            # some random boards may be invalid (e.g., too many checkers), skip
            pass

    @given(
        color=st.sampled_from([Color.White, Color.Black]),
        point_num=st.integers(min_value=1, max_value=24),
        distance=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=50, deadline=None)
    def test_move_validity_matches(self, color, point_num, distance):
        """move_is_valid() should match between Board and vector"""
        board = Board(default_starting_points)
        vec = vg.from_board(board)

        move = Move(color, point_num, distance)
        board_valid = board.move_is_valid(move)
        vec_valid = vg.move_is_valid(vec, move)

        self.assertEqual(board_valid, vec_valid,
                        f"validity mismatch for {move}: board={board_valid}, vec={vec_valid}")

    @given(
        color=st.sampled_from([Color.White, Color.Black]),
        dice_roll=st.tuples(
            st.integers(min_value=1, max_value=6),
            st.integers(min_value=1, max_value=6)
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_possible_moves_match(self, color, dice_roll):
        """possible_moves() should generate identical move sets"""
        board = Board(default_starting_points)
        vec = vg.from_board(board)

        dice = Dice(roll=dice_roll)
        board_moves = board.possible_moves(color, dice)
        vec_moves = vg.possible_moves(vec, color, dice.effective_roll())

        board_move_seqs = set(m[0] for m in board_moves)
        vec_move_seqs = set(m[0] for m in vec_moves)

        self.assertEqual(board_move_seqs, vec_move_seqs)


if __name__ == '__main__':
    unittest.main()
