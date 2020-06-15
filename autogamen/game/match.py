from collections import Counter

from .game import Game
from .types import Color

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

  def start_game(self):
    self.current_game = Game(self.players)
    self.games.append(self.current_game)
    self.current_game.start()

  def tick(self):
    """Returns boolean whether the current game is over.
    """
    self.current_game.run_turn()

    if self.current_game.winner:
      self.points[self.current_game.winner.color] += self.current_game.points
      self.turn_count += self.current_game.turn_number

      if self.points[self.current_game.winner.color] >= self.point_goal:
        self.winner = self.current_game.winner

      return True

    return False
