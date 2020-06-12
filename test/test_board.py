from autogamen.game.board import Board
from autogamen.game.types import Color, Point

import unittest

default_starting_board = [
  Point(2, Color.White),
  Point(0),
  Point(0),
  Point(0),
  Point(0),
  Point(5, Color.Black),
  Point(0),
  Point(3, Color.Black),
  Point(0),
  Point(0),
  Point(0),
  Point(5, Color.White),
  Point(5, Color.Black),
  Point(0),
  Point(0),
  Point(0),
  Point(3, Color.White),
  Point(0),
  Point(5, Color.White),
  Point(0),
  Point(0),
  Point(0),
  Point(0),
  Point(2, Color.Black),
]

class TestBoard(unittest.TestCase):
  def test_constructor_invalid_inputs(self):
    with self.assertRaises(Exception, msg="Raises for empty board"):
      Board([])

    with self.assertRaises(Exception, msg="Raises for too-large board"):
      Board(default_starting_board + [Point(2, Color.Black)])

    white_imbalanced_points = list(default_starting_board)
    white_imbalanced_points[0] = Point(4, Color.White)
    with self.assertRaises(Exception, msg="Raises for imbalanced pips"):
      Board(white_imbalanced_points)


  def test_can_bear_off_starting(self):
    board = Board(default_starting_board)
    self.assertFalse(board.can_bear_off(Color.White))
    self.assertFalse(board.can_bear_off(Color.Black))

  def test_can_bear_off_starting(self):
    board = Board(default_starting_board)
    self.assertFalse(board.can_bear_off(Color.White))
    self.assertFalse(board.can_bear_off(Color.Black))
