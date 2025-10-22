
from typing import Any

from .game import Game
from .game_types import Color


class Match:
  def __init__(self, players: Any, point_goal: int) -> None:
    self.players = players
    self.point_goal = point_goal

    # Set on the conclusion of each game.
    self.current_game: Game | None = None
    self.games: list[Game] = []
    self.points = {Color.White: 0, Color.Black: 0}
    self.turn_count = 0

    # Set after a winner is decided
    self.winner: Any | None = None

    self.tick_prepared = False

  def pre_tick(self) -> None:
    if not self.winner:
      if not self.current_game:
        self.current_game = Game(self.players)
        self.games.append(self.current_game)
        self.current_game.start()
      else:
        self.current_game.pre_turn()
      self.tick_prepared = True

  def tick(self) -> bool:
    """Returns boolean whether the current game is over.
    """
    if self.winner:
      return True

    if not self.tick_prepared:
      self.pre_tick()
      self.tick_prepared = False

    assert self.current_game is not None
    self.current_game.turn_blocking()

    winner = self.current_game.winner
    if not winner:
      return False

    self.points[winner.color] += self.current_game.points  # type: ignore[unreachable]
    self.turn_count += self.current_game.turn_number

    if self.points[winner.color] >= self.point_goal:
      self.winner = winner

    return True
