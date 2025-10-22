"""Player implementations for backgammon AI."""
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from autogamen.game.game_types import TurnAction

if TYPE_CHECKING:
    from autogamen.game.board import _Board
    from autogamen.game.game import Game
    from autogamen.game.game_types import Color


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
