
from .game import Game
from .game_types import Color


class Match:
  def __init__(self, players, point_goal):
    self.players = players
    self.point_goal = point_goal

    # Set on the conclusion of each game.
    self.current_game = None
    self.games = []
    self.points = {Color.White: 0, Color.Black: 0}
    self.turn_count = 0

    # Set after a winner is decided
    self.winner = None

    self.tick_prepared = False

  def pre_tick(self):
    if not self.winner:
      if not self.current_game:
        self.current_game = Game(self.players)
        self.games.append(self.current_game)
        self.current_game.start()
      else:
        self.current_game.pre_turn()
      self.tick_prepared = True

  def tick(self):
    """Returns boolean whether the current game is over.
    """
    if self.winner:
      return True

    if not self.tick_prepared:
      self.pre_tick()
      self.tick_prepared = False

    self.current_game.turn_blocking()

    if self.current_game.winner:
      self.points[self.current_game.winner.color] += self.current_game.points
      self.turn_count += self.current_game.turn_number

      if self.points[self.current_game.winner.color] >= self.point_goal:
        self.winner = self.current_game.winner

      return True

    return False
