from typing import TYPE_CHECKING

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

  def action(self, possible_moves: set[tuple[tuple, "_Board"]]) -> list:
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
