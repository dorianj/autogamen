from collections import defaultdict
import copy
import itertools

from .types import Color, Move

class Board:
  def __init__(self, points, bar=None, off=None):
    if len(points) != 24:
      raise Exception(f"{len(points)} board passed in.")

    self.points = points
    self.bar = bar or {Color.White: 0, Color.Black: 0}
    self.off = off or {Color.White: 0, Color.Black: 0}

  def can_bear_off(self, color):
    if self.bar[color] != 0:
      return False

    home_range = self.home_board_range(color)

    for i, point in enumerate(self.points):
      point_number = i + 1
      if not (home_range[0] <= point_number <= home_range[1]):
        if color == point.color:
          return False

    return True

  def add_off(self, color):
    self.off[color] += 1

  def winner(self):
    """Returns the winning player's color, or None if no one has won
    """
    for color, pips in self.off.items():
      if pips == 15:
        return color

    return None

  def winner_stakes(self):
    """Returns 3 for a backgammon, 2 for a gammon, and 1 for a regular win.
    """
    winner = self.winner()
    if not winner:
      raise Exception("winner_stakes isn't valid unless a win occurs")

    opponent = winner.opponent()
    opponent_pips_off = self.off[opponent]
    opponent_pips_on_bar = self.bar[opponent]
    oppponen_pips_in_winner_home = sum(
      self.point_at_number(point_number).count
      for point_number in range(*self.home_board_range(winner)))

    if opponent_pips_off > 0:
      # Normal win, if the oppoennt has borne off any pips.
      return 1
    elif (opponent_pips_on_bar + oppponen_pips_in_winner_home) != 0:
      # If the opponent has not yet borne off any checkers and has some on the bar
      # or in the winner's home board, the winner scores a backgammon.
      return 3
    else:
      # If the opponent has not yet borne off any checkers.
      return 2

  def home_board_range(self, color):
    if color == Color.Black:
      return [1, 6]
    else:
      return [19, 24]

  def point_at_number(self, point_number):
    if point_number == 0:
      raise Exception("Board.point_at_number attempted to get the bar")

    return self.points[point_number - 1]

  def move_is_valid(self, move):
    """Returns whether a single move is allowed, i.e. that if it would bear off,
       the player is currently allowed to bear off, that it doesn't try to hit
       a block, that it is coming from a non-empty space, etc.
    """

    # These may be incalcuable, so lazily compute them
    destination_point = lambda: self.point_at_number(move.destination_point_number())
    source_point = lambda: self.point_at_number(move.point_number)

    # This is coming off of the bar.
    if move.point_number == Move.Bar:
      return self.bar[move.color] > 0 and destination_point().can_land(move.color)

    # If this color has anything on the bar, can't move on the board
    if self.bar[move.color] > 0:
      return False

    # Must have at least one of own color to move
    if source_point().is_empty() or source_point().color != move.color:
      return False
    elif move.destination_is_off():
      return self.can_bear_off(move.color)
    else:
      return destination_point().can_land(move.color)

  def add_bar(self, color):
    self.bar[color] += 1

  def subtract_bar(self, color):
    if self.bar[color] < 1:
      raise Exception("Board.subtract_bar: bar is empty")

    self.bar[color] -= 1

  def apply_move(self, move):
    """Mutate this board to apply :move:
    """
    if not self.move_is_valid(move):
      raise Exception("Invalid move passed to apply_move")

    # Subtract the source pip, wherever it may be
    if move.point_number == Move.Bar:
      self.subtract_bar(move.color)
    else:
      source_point = self.point_at_number(move.point_number)
      source_point.subtract(move.color)

    # Move this pip, either off or on the board
    if move.destination_is_off():
      self.add_off(move.color)
    else:
      destination_point = self.point_at_number(move.destination_point_number())

      if destination_point.can_hit(move.color):
        destination_point.hit(move.color)
        self.add_bar(move.color.opponent())
      else:
        destination_point.add(move.color)

  def possible_moves(self, color, dice):
    """Returns the list of possible moves. Each item is a Set of Moves
    """
    def _worker(board, remaining_dice):
      if not len(remaining_dice):
        return []

      moves = []
      for d, die in enumerate(set(remaining_dice)):
        # Attempt to find moves on the board or bar. We're gonna do a little
        # sneaky thing and start at 0, which is Move.Bar, which will ensure that
        # we get pips off the bar before moving pips on the board. I'm sorry.
        # This is horrendously clever (and not even that smart), but it did
        # save a good bit of lines of code.
        for point_number in range(0, 25):
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
