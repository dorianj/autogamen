"""Player implementations for backgammon AI."""
import random
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from autogamen.game.game_types import Color, Move, TurnAction

if TYPE_CHECKING:
    from autogamen.game.board import _Board
    from autogamen.game.game import Game
    from autogamen.gnubg.interface import GnubgInterface


class Player:
  def __init__(self, color: "Color") -> None:
    self.color = color
    self.game: Game | None = None

  def start_game(self, game: "Game") -> None:
    self.game = game

  def action(self, possible_moves: set[tuple[tuple[Any, ...], "_Board"]]) -> list[Any]:
    """Called at the start of a turn.
    Return: [TurnAction, moves?]
    """
    raise Exception("action not implemented")

  def accept_doubling_cube(self) -> bool:
    """Called when opponent has offered the doubling cube.
    Return: boolean, true to accept; false to forfeit
    """
    raise Exception("doubling_cube not implemented")

  def end_game(self, game: "Game") -> None:
    self.game = None


class BozoPlayer(Player):
  """Picks a random move every time.
  """
  def action(self, possible_moves: set[tuple[tuple[Any, ...], Any]]) -> list[Any]:
    if not len(possible_moves):
      return [TurnAction.Pass]

    return [TurnAction.Move, random.choice(sorted(possible_moves))[0]]

  def accept_doubling_cube(self) -> bool:
    return False



class RunningPlayer(Player):
  """Picks the move that minimizes own pip count.
  """
  def action(self, possible_moves: set[tuple[tuple[Any, ...], Any]]) -> list[Any]:
    if not len(possible_moves):
      return [TurnAction.Pass]

    boards_by_pip_count: defaultdict[Any, set[Any]] = defaultdict(set)
    for moves, board in possible_moves:
      boards_by_pip_count[board.pip_count()[self.color]].add(moves)

    possible_moves = boards_by_pip_count[min(boards_by_pip_count.keys())]
    return [TurnAction.Move, random.choice(sorted(possible_moves))]

  def accept_doubling_cube(self) -> bool:
    return True


class DeltaPlayer(Player):
  """Picks the move that maximizes pip count delta.
  """
  def action(self, possible_moves: set[tuple[tuple[Any, ...], Any]]) -> list[Any]:
    if not len(possible_moves):
      return [TurnAction.Pass]

    boards_by_pip_delta: defaultdict[Any, set[Any]] = defaultdict(set)
    for moves, board in possible_moves:
      pip_delta = board.pip_count()[self.color.opponent()] - board.pip_count()[self.color]
      boards_by_pip_delta[pip_delta].add(moves)

    possible_moves = boards_by_pip_delta[max(boards_by_pip_delta.keys())]
    return [TurnAction.Move, random.choice(sorted(possible_moves))]

  def accept_doubling_cube(self) -> bool:
    return True


class GnubgPlayer(Player):
  """uses gnubg AI to select moves."""

  gnubg: "GnubgInterface"

  def __init__(self, color: Color, plies: int = 2) -> None:
    super().__init__(color)
    # local import prevents circular dependency
    from autogamen.gnubg.interface import GnubgInterface  # noqa: PLC0415
    self.gnubg = GnubgInterface(plies=plies)
    self.gnubg.start()

  def action(self, possible_moves: set[tuple[tuple[Any, ...], Any]]) -> list[Any]:
    if not len(possible_moves):
      return [TurnAction.Pass]

    # get the current board and dice from game
    if self.game is None:
      raise RuntimeError("GnubgPlayer.action called without game")

    board = self.game.board
    dice = self.game.active_dice

    if dice is None:
      raise RuntimeError("GnubgPlayer.action called without active dice")

    # get gnubg's suggestion
    hints = self.gnubg.get_hint(board, self.color, dice.roll)

    if not hints:
      # gnubg didn't return any hints, fall back to random
      return [TurnAction.Move, random.choice(sorted(possible_moves))[0]]

    # parse gnubg's best move and convert to our move format
    best_gnubg_move = hints[0].moves

    # convert gnubg move string (e.g. "24/18 13/11") to our Move objects
    try:
      our_moves = self._parse_gnubg_move_string(best_gnubg_move)

      # find the matching move in possible_moves
      for moves, _ in possible_moves:
        if self._moves_match(moves, our_moves):
          return [TurnAction.Move, moves]

      # if we couldn't find a match, fall back to random
      return [TurnAction.Move, random.choice(sorted(possible_moves))[0]]

    except Exception:
      # if parsing fails, fall back to random
      return [TurnAction.Move, random.choice(sorted(possible_moves))[0]]

  def _gnubg_to_our_point(self, gnubg_point: int) -> int:
    """convert gnubg's point numbering to our internal point numbering.

    gnubg uses player-relative numbering:
    - for white: gnubg point 24 = our point 1, gnubg point 1 = our point 24
    - for black: gnubg point 24 = our point 24, gnubg point 1 = our point 1

    so for white, we need to flip: our_point = 25 - gnubg_point
    for black, it's the same: our_point = gnubg_point
    """
    if self.color == Color.White:
      return 25 - gnubg_point
    else:
      return gnubg_point

  def _parse_gnubg_move_string(self, move_str: str) -> tuple[Move, ...]:
    """convert gnubg move notation like "24/18 13/11" to our Move objects.

    gnubg uses player-relative notation where point 24 is always the starting position
    and point 1 is near the bear-off area, regardless of player color.
    """
    move_parts = move_str.split()
    moves = []

    for part in move_parts:
      match = re.match(r'(\d+|bar)/(\d+|off)', part.lower())
      if match:
        source_str = match.group(1)
        dest_str = match.group(2)

        if source_str == "bar":
          our_source = Move.Bar
        else:
          gnubg_source = int(source_str)
          our_source = self._gnubg_to_our_point(gnubg_source)

        if dest_str == "off":
          # bearing off
          if our_source == Move.Bar:
            raise ValueError("cannot bear off from bar")
          if self.color == Color.White:
            distance = our_source
          else:
            distance = 25 - our_source
        else:
          gnubg_dest = int(dest_str)
          our_dest = self._gnubg_to_our_point(gnubg_dest)
          distance = abs(our_source - our_dest)

        moves.append(Move(self.color, our_source, distance))

    return tuple(moves)

  def _moves_match(self, moves1: tuple[Any, ...], moves2: tuple[Move, ...]) -> bool:
    """check if two move sequences are equivalent."""
    if len(moves1) != len(moves2):
      return False

    # moves can be in different order, so compare as sets
    return set(moves1) == set(moves2)

  def accept_doubling_cube(self) -> bool:
    # for now, always accept
    return True

  def end_game(self, game: "Game") -> None:
    super().end_game(game)

  def __del__(self) -> None:
    if hasattr(self, 'gnubg'):
      self.gnubg.stop()
