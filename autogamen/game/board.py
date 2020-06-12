from collections import defaultdict
import copy
import itertools

from .types import Color, Move

class Board:
  def __init__(self, points):
    if len(points) != 24:
      raise Exception(f"{len(points)} board passed in.")

    self.points = points

  def is_complete(self):
    """Returns boolean indicating whether all pips are accounted for.
    """
    counts = defaultdict(int)
    for point in self.points:
      if point.color is not None:
        counts[point.color] += point.count

    # TODO: this doesn't cover points that are off the board
    return list(counts.values()) == [15, 15]

  def can_bear_off(self, color):
    home_range = self.home_board_range(color)

    for i, point in enumerate(self.points):
      point_number = i + 1
      if not (home_range[0] <= point_number <= home_range[1]):
        if color == point.color:
          return False

    return True

  def home_board_range(self, color):
    if color == Color.Black:
      return [1, 6]
    else:
      return [19, 24]

  def point_at_number(self, point_number):
    return self.points[point_number - 1]

  def move_is_valid(self, move):
    source_point = self.point_at_number(move.point_number)

    # Must have at least one of own color to move
    if source_point.is_empty() or source_point.color != move.color:
      return False

    if move.destination_is_bar():
      # This would bar off
      return self.can_bear_off(move.color)

    destination_point = self.point_at_number(move.destination_point_number())
    return destination_point.can_add(move.color)

  def apply_move(self, move):
    if not self.move_is_valid(move):
      raise Exception("Invalid move passed to apply_move")

    source_point = self.point_at_number(move.point_number)
    source_point.subtract(move.color)

    if move.destination_is_bar():
      #raise Exception("TODO: Board.apply_move to the bar")
      return

    destination_point = self.point_at_number(move.destination_point_number())

    if destination_point.can_hit(move.color):
      destination_point.hit(move.color)
      # TODO: put the opposing pip onto the bar
    else:
      destination_point.add(move.color)

  def possible_moves(self, color, dice):
    """Returns the list of possible moves. Each item is a Set of Moves
    """

    def _worker(board, remaining_dice):
      if not len(remaining_dice):
        return []

      moves = []
      for d, die in enumerate(remaining_dice):
        for i, point in enumerate(board.points):
          point_number = i + 1

          move = Move(color, point_number, die)
          if board.move_is_valid(move):
            moves.append((move,))
            new_board = copy.deepcopy(board)
            new_board.apply_move(move)
            for submoves in _worker(new_board, remaining_dice[0:d] + remaining_dice[d + 1:]):
              moves.append((move,) + submoves)

      return tuple(moves)


    effective_roll = dice.effective_roll()
    all_movesets = _worker(self, effective_roll)

    if len(all_movesets) == 0:
      return set()

    # Prune incomplete moves. Basically, if there exists movesets that use all
    # of the dice, then the ones that only use some of the dice are invalid.
    longest_moveset = max(map(len, all_movesets))
    if longest_moveset > len(effective_roll):
      raise Exception("Logic error: shouldn't have movesets with more moves than dice")

    return set(moveset for moveset in all_movesets if len(moveset) == longest_moveset)
