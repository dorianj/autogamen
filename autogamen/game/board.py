from collections import Counter, defaultdict
import itertools

from .types import Color, FrozenPoint, Move

class _Board:
  def __init__(self, points, bar=None, off=None):
    if len(points) != 24:
      raise Exception(f"{len(points)} board passed in.")

    self.points = tuple(point.mutable_copy() for point in points)
    self.bar = list(bar or self.color_counter())
    self.off = list(off or self.color_counter())

  def __eq__(self, other):
    return (
      all(self.bar[color.value] == other.bar[color.value] and
          self.off[color.value] == other.off[color.value]
          for color in Color) and
      self.points == other.points
    )

  def frozen_copy(self):
    """Returns an immutable FrozenBoard copy of this object.
    """
    if isinstance(self, FrozenBoard):
      return self

    return FrozenBoard(self.points, self.bar, self.off)

  def mutable_copy(self):
    """Returns a mutable Board copy of this object.
    """
    return Board(self.points, self.bar, self.off)

  @staticmethod
  def color_counter(count_dict=None):
    """Returns an object representing off or bar
    """
    if count_dict is None:
      return [0, 0]
    else:
      c = [0, 0]
      for color in Color:
        c[color.value] = count_dict.get(color, 0)
      return c

  def visual_str_repr(self):
    rows = ["", "", ""]
    height = 5
    columns = []
    for point_number in range(1, 25):
      point = self.point_at_number(point_number)
      rows[0] += (f"{point.count:>2} ")
      rows[1] += " " + ("B" if point.color == Color.Black else "W") + " "
      rows[2] += (f"{point_number:>2} ")

    rows.append(f"Bar: W:{self.bar[Color.White.value]}, B:{self.bar[Color.Black.value]}")
    rows.append(f"Off: W:{self.off[Color.White.value]}, B:{self.off[Color.Black.value]}")

    return "\n".join(rows)


  def pip_count(self):
    """Returns a dict of pip count (sum of distance from fully beared off) for each player
    """
    counter = Counter()
    for point_number in range(1,25):
      point = self.point_at_number(point_number)
      if point.color is not None:
        distance = 25 - point_number if point.color == Color.White else point_number
        counter[point.color] += point.count * distance

    for color in Color:
      counter[color] += self.bar[color.value] * 24

    return counter

  def can_bear_off(self, color):
    if self.bar[color.value] != 0:
      return False

    home_range = self.home_board_range(color)

    for i, point in enumerate(self.points):
      point_number = i + 1
      if not (home_range[0] <= point_number <= home_range[1]):
        if color == point.color:
          return False

    return True

  def winner(self):
    """Returns the winning player's color, or None if no one has won
    """
    for color in Color:
      if self.off[color.value] == 15:
        return color

    return None

  def winner_stakes(self):
    """Returns 3 for a backgammon, 2 for a gammon, and 1 for a regular win.
    """
    winner = self.winner()
    if not winner:
      raise Exception("winner_stakes isn't valid unless a win occurs")

    opponent = winner.opponent()
    opponent_pips_off = self.off[opponent.value]
    opponent_pips_on_bar = self.bar[opponent.value]
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

  def copy_apply_moves(self, moves):
    """Returns a frozen copy of self with :moves: applied. Less readable than
       Board.apply_move, but more performant. Note that all moves must have the
       same color, if that invariant isn't held behavior is undefined.
    """
    new_bar = list(self.bar)
    new_off = list(self.off)
    new_points = list(self.points)

    for move in moves:
      # Subtract the source pip, wherever it may be
      if move.point_number == Move.Bar:
        new_bar[move.color.value] -= 1
      else:
        source_point = new_points[move.point_number - 1]
        if isinstance(source_point, FrozenPoint):
          new_points[move.point_number - 1] = source_point = source_point.mutable_copy()

        source_point.subtract(move.color)

      # Move this pip, either off or on the board
      if move.destination_is_off:
        new_off[move.color.value] += 1
      else:
        dest_point = new_points[move.destination_point_number - 1]
        if isinstance(dest_point, FrozenPoint):
          new_points[move.destination_point_number - 1] = dest_point = dest_point.mutable_copy()

        if dest_point.can_hit(move.color):
          new_bar[move.color.opponent().value] += 1
          dest_point.hit(move.color)
        else:
          dest_point.add(move.color)

    return FrozenBoard(new_points, new_bar, new_off)

  def move_is_valid(self, move):
    """Returns whether a single move is allowed, i.e. that if it would bear off,
       the player is currently allowed to bear off, that it doesn't try to hit
       a block, that it is coming from a non-empty space, etc.
    """

    # This may be incalcuable, so lazily compute it
    destination_point = lambda: self.point_at_number(move.destination_point_number)

    # This is coming off of the bar.
    if move.point_number == Move.Bar:
      return self.bar[move.color.value] > 0 and destination_point().can_land(move.color)

    # If this color has anything on the bar, can't move on the board
    if self.bar[move.color.value] > 0:
      return False

    source_point = self.point_at_number(move.point_number)
    if source_point.color != move.color or source_point.count == 0:
      return False
    elif move.destination_is_off:
      return self.can_bear_off(move.color)
    else:
      return destination_point().can_land(move.color)

  def possible_moves(self, color, dice):
    """Returns the list of possible moves. Each item is a Set of Moves
    """
    def _worker(board, remaining_dice):
      moves = set()
      for d, die in enumerate(set(remaining_dice)):
        # Attempt to find moves on the board or bar. We're gonna do a little
        # sneaky thing and start at 0, which is Move.Bar, which will ensure that
        # we get pips off the bar before moving pips on the board. I'm sorry.
        # This is horrendously clever (and not even that smart), but it did
        # save a good bit of lines of code.
        for point_number in range(0, 25):
          move = Move(color, point_number, die)
          if board.move_is_valid(move):
            new_board = board.copy_apply_moves([move]).frozen_copy()
            moves.add(((move,), new_board))
            for submoves, subboard in _worker(new_board, remaining_dice[0:d] + remaining_dice[d + 1:]):
              moves.add(((move,) + submoves, subboard))

      return moves

    effective_roll = dice.effective_roll()
    all_moves = _worker(self.frozen_copy(), effective_roll)

    if len(all_moves) == 0:
      return set()

    # Prune incomplete moves. Basically, if there exists movesets that use all
    # of the dice, then the ones that only use some of the dice are invalid.
    longest_move = max(map(lambda m: len(m[0]), all_moves))
    if longest_move > len(effective_roll):
      raise Exception(f"Logic error: shouldn't have movesets with more moves than dice ({longest_move} > {len(effective_roll)})")

    return set(m for m in all_moves if len(m[0]) == longest_move)


class FrozenBoard(_Board):
  """An Immutable Board.
  """
  def __init__(self, points, bar=None, off=None):
    if len(points) != 24:
      raise Exception(f"{len(points)} board passed in.")

    self.points = tuple(point.frozen_copy() for point in points)
    self.bar = tuple(bar or Board.color_counter())
    self.off = tuple(off or Board.color_counter())
    self._hash = hash((self.points, self.bar, self.off))

  def __hash__(self):
    return self._hash


class Board(_Board):
  def add_off(self, color):
    self.off[color.value] += 1

  def add_bar(self, color):
    self.bar[color.value] += 1

  def subtract_bar(self, color):
    if self.bar[color.value] < 1:
      raise Exception("Board.subtract_bar: bar is empty")

    self.bar[color.value] -= 1

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
    if move.destination_is_off:
      self.add_off(move.color)
    else:
      destination_point = self.point_at_number(move.destination_point_number)

      if destination_point.can_hit(move.color):
        destination_point.hit(move.color)
        self.add_bar(move.color.opponent())
      else:
        destination_point.add(move.color)

