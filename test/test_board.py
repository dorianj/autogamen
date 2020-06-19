import unittest
import itertools

from autogamen.game.board import Board, FrozenBoard
from autogamen.game.types import Color, Dice, Move, Point
from autogamen.ui.match_view import display_board

from .performance import assertRuntime

def repeat_point(repeat_count, pips=0, color=None):
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

white_can_bear_off_points = [
  Point(5, Color.Black),
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
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(7, Color.White),
  Point(2, Color.White),
  Point(2, Color.White),
  Point(2, Color.White),
  Point(2, Color.White),
  Point(2, Color.Black),
]

white_perfect_prime_points = [
  Point(5, Color.Black),
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
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(2, Color.White),
  Point(5, Color.White),
  Point(2, Color.White),
  Point(2, Color.White),
  Point(2, Color.White),
  Point(2, Color.White),
  Point(2, Color.Black),
]

white_almost_done_points = [
  Point(5, Color.Black),
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
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(),
  Point(2, Color.White),
  Point(),
  Point(),
  Point(),
  Point(2, Color.White),
  Point(2, Color.Black),
]


def extract_moves(moves_and_boards):
  return set(m[0] for m in moves_and_boards)


class TestBoardValidation(unittest.TestCase):
  def test_constructor_invalid_inputs(self):
    with self.assertRaises(Exception, msg="Raises for empty board"):
      Board([])

    with self.assertRaises(Exception, msg="Raises for too-large board"):
      Board(default_starting_points + [Point(2, Color.Black)])


class TestFrozenBoard(unittest.TestCase):
  def test_board_equality(self):
    b1 = Board(list(default_starting_points))
    b2 = FrozenBoard(list(default_starting_points))
    self.assertEqual(b1, b1)

  def test_frozen_board_hashable(self):
    b1 = Board(list(default_starting_points))
    b1.add_off(Color.White)
    b1.add_bar(Color.Black)
    b2 = Board(list(default_starting_points))
    b2.add_off(Color.White)
    b2.add_bar(Color.Black)
    self.assertEqual(hash(b1.frozen_copy()), hash(b2.frozen_copy()))


class TestBoardBearOff(unittest.TestCase):
  def test_can_bear_off_starting(self):
    board = Board(default_starting_points)
    self.assertFalse(board.can_bear_off(Color.White))
    self.assertFalse(board.can_bear_off(Color.Black))

  def test_can_bear_off_limited(self):
    board = Board(white_can_bear_off_points)
    self.assertTrue(board.can_bear_off(Color.White))
    self.assertFalse(board.can_bear_off(Color.Black))

  def test_can_bear_off_perfect_prime(self):
    board = Board(white_perfect_prime_points)
    self.assertFalse(board.can_bear_off(Color.White))
    self.assertFalse(board.can_bear_off(Color.Black))

  def test_can_bear_off_almost_done(self):
    board = Board(white_almost_done_points)
    self.assertTrue(board.can_bear_off(Color.White))
    self.assertFalse(board.can_bear_off(Color.Black))


class TestBoardMoves(unittest.TestCase):
  def test_single_pip_anywhere(self):
    board = Board(
      repeat_point(10) +
      [Point(1, Color.White)] +
      repeat_point(13)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,2)))),
      set([
        (Move(Color.White, 11, 1), Move(Color.White, 12, 2)),
        (Move(Color.White, 11, 2), Move(Color.White, 13, 1)),
      ])
    )

  def test_single_pip_anywhere(self):
    board = Board(
      repeat_point(10) +
      [Point(1, Color.White)] +
      repeat_point(13)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,2)))),
      set([
        (Move(Color.White, 11, 1), Move(Color.White, 12, 2)),
        (Move(Color.White, 11, 2), Move(Color.White, 13, 1)),
      ])
    )

  def test_doubles(self):
    board = Board(
      repeat_point(10) +
      [Point(1, Color.White)] +
      repeat_point(13)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(2,2)))),
      set([
        (Move(Color.White, 11, 2), Move(Color.White, 13, 2),
         Move(Color.White, 15, 2), Move(Color.White, 17, 2)),
      ])
    )

  def test_hit_opportunity(self):
    board = Board(
      repeat_point(10) +
      [Point(1, Color.White)] +
      [Point(1, Color.Black)] +
      repeat_point(12)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,2)))),
      set([
        (Move(Color.White, 11, 1), Move(Color.White, 12, 2)),
        (Move(Color.White, 11, 2), Move(Color.White, 13, 1)),
      ])
    )

  def test_single_block(self):
    board = Board(
      repeat_point(10) +
      [Point(1, Color.White)] +
      [Point(2, Color.Black)] +
      repeat_point(12)
    )

    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,2)))),
      set([
        (Move(Color.White, 11, 2), Move(Color.White, 13, 1)),
      ])
    )

  def test_single_valid_move(self):
    board = Board(
      repeat_point(10) +
      [Point(1, Color.White)] +
      [Point(2, Color.White)] +
      repeat_point(12, Color.Black, 2)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,3)))),
      set([
        (Move(Color.White, 11, 1),),
      ])
    )

  # if a player rolls a 6 and a 5, but has no checkers on the 6-point and two
  # on the 5-point, then the 6 and the 5 must be used to bear off the two
  # checkers from the 5-point
  def test_two_pip_bear_off(self):
    board = Board(
      repeat_point(19) +
      [Point(2, Color.White)] +
      repeat_point(4)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(5,6)))),
      set([
        # These  movesets are identical to each other because they both bear
        # off and could be deduped, but don't seem harmful, so leaving for now
        (Move(Color.White, 20, 6), Move(Color.White, 20, 5),),
        (Move(Color.White, 20, 5), Move(Color.White, 20, 6),),
      ])
    )

  # f a player has exactly one checker remaining on the 6-point, and rolls a
  # 6 and a 1, the player may move the 6-point checker one place to the
  # 5-point with the lower die roll of 1, and then bear that checker off the
  # 5-point using the die roll of 6
  def test_one_pip_bear_off(self):
    board = Board(
      repeat_point(5) +
      [Point(1, Color.Black)] +
      repeat_point(18)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.Black, Dice(roll=(1,6)))),
      set([
        (Move(Color.Black, 6, 1), Move(Color.Black, 5, 6),),
        # Practically, this move is idenical since the above two
        # moves result in bearing off. However the above move uses
        # both dice, and so is considered correct
        # (Move(Color.Black, 6, 6),),
      ])
    )

  def test_one_hit_pip_bear_off(self):
    """Sames as test_one_pip_bear_off but results in a hit"""
    board = Board(
      repeat_point(4) +
      [Point(1, Color.White)] +
      [Point(1, Color.Black)] +
      repeat_point(18)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.Black, Dice(roll=(1,6)))),
      set([
        (Move(Color.Black, 6, 1), Move(Color.Black, 5, 6),),
      ])
    )

  def test_one_blocked_pip_bear_off(self):
    """Sames as test_one_pip_bear_off but 1 is blocked"""
    board = Board(
      repeat_point(4) +
      [Point(2, Color.White)] +
      [Point(1, Color.Black)] +
      repeat_point(18)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.Black, Dice(roll=(1,6)))),
      set([
        (Move(Color.Black, 6, 6),),
      ])
    )

  def test_perfect_prime_no_moves(self):
    board = Board(
      [Point(2, Color.White)] +
      repeat_point(6, 2, Color.Black) +
      repeat_point(17)
    )
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(6,6)))),
      set()
    )

  def test_can_get_off_bar_with_empty_home(self):
    board = Board(
      repeat_point(24)
    )
    board.add_bar(Color.White)
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,6)))),
      set([
        (Move(Color.White, Move.Bar, 1), Move(Color.White, 1, 6),),
        (Move(Color.White, Move.Bar, 6), Move(Color.White, 6, 1),),
      ])
    )

  def test_can_get_off_bar_with_parly_full_home(self):
    board = Board(
      repeat_point(5, 2, Color.Black) +
      repeat_point(19)
    )
    board.add_bar(Color.White)
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,6)))),
      set([
        (Move(Color.White, Move.Bar, 6), Move(Color.White, 6, 1),),
      ])
    )

  def test_cant_get_off_bar_with_full_home(self):
    board = Board(
      repeat_point(6, 2, Color.Black) +
      repeat_point(18)
    )
    board.add_bar(Color.White)
    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,6)))),
      set()
    )

  def test_white_can_bear_off_from_24(self):
    board = Board(
      repeat_point(23) +
      repeat_point(1, 1, Color.White)
    )

    self.assertEqual(
      extract_moves(board.possible_moves(Color.White, Dice(roll=(1,6)))),
      set([
        (Move(Color.White, 24, 1),),
        (Move(Color.White, 24, 6),),
      ])
    )


class TestBar(unittest.TestCase):
  def test_cant_bear_off_with_bar(self):
    board = Board(
      repeat_point(23) +
      [Point(2, Color.White)],
    )
    board.add_bar(Color.White)
    self.assertFalse(board.can_bear_off(Color.White))


class TestApplyMoves(unittest.TestCase):
  def test_simple_move(self):
    board = Board(
      [Point(2, Color.White)] +
      repeat_point(23)
    )
    board.apply_move(Move(Color.White, 1, 6))
    self.assertEqual(board.point_at_number(1), Point(1, Color.White))
    self.assertEqual(board.point_at_number(7), Point(1, Color.White))

  def test_simple_hit(self):
    board = Board(
      [Point(2, Color.White)] +
      [Point(1, Color.Black)] +
      repeat_point(22)
    )
    board.apply_move(Move(Color.White, 1, 1))
    self.assertEqual(board.bar[Color.Black], 1)

  def test_bear_off_and_hit(self):
    board = Board(
      repeat_point(22) +
      [Point(1, Color.White)] +
      [Point(1, Color.Black)]
    )
    board.apply_move(Move(Color.White, 23, 1))
    self.assertEqual(board.bar[Color.Black], 1)
    board.apply_move(Move(Color.White, 24, 1))

    self.assertEqual(board.off[Color.White], 1)

class TestPipCount(unittest.TestCase):
  def test_single_white_pip(self):
    board = Board(
      repeat_point(1, 1, Color.White) +
      repeat_point(23)
    )
    self.assertEqual(board.pip_count(), {
      Color.Black: 0,
      Color.White: 24
    })

  def test_single_black_pip(self):
    board = Board(
      repeat_point(1, 1, Color.Black) +
      repeat_point(23)
    )
    self.assertEqual(board.pip_count(), {
      Color.Black: 1,
      Color.White: 0,
    })

  def test_single_barred_pip(self):
    board = Board(
      repeat_point(24)
    )
    board.add_bar(Color.White)
    self.assertEqual(board.pip_count(), {
      Color.Black: 0,
      Color.White: 24,
    })

  def test_starting_board(self):
    board = Board(default_starting_points)
    self.assertEqual(board.pip_count(), {
      Color.Black: 167,
      Color.White: 167,
    })

class TestPerformance(unittest.TestCase):
  def test_double_roll_filled_board_performance(self):
    board = Board(
      repeat_point(2, 1, Color.White) +
      repeat_point(2, 1, Color.Black) +
      repeat_point(20)
    )
    with assertRuntime(self, 0.1):
      board.possible_moves(Color.White, Dice(roll=[2,2]))
