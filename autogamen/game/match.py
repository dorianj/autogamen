from collections import Counter

from .game import Game
from .types import Color

class Match:
  def __init__(self, players, point_goal):
    self.players = players
    self.point_goal = point_goal

    self.games = []
    self.points = {Color.White: 0, Color.Black: 0}

  def start(self):
    self.start_new_game()

  def start_new_game(self):
    self.current_game = Game(self.players)
    self.games.append(self.current_game)
    self.current_game.start()

  def tick(self):
    self.current_game.run_turn()

    if self.current_game.winner:
      turn_count = sum(game.turn_number for game in self.games)
      print(f"Game ended! {self.current_game.winner.color} won with {self.current_game.points} points after {turn_count} turns")
      self.points[self.current_game.winner.color] += self.current_game.points

      if self.points[self.current_game.winner.color] >= self.point_goal:
        game_count = len(self.games)
        print(f"Match ended! {self.current_game.winner.color} won with {self.points[self.current_game.winner.color]} points in {game_count} games")
        for color, win_count in Counter(game.winner.color for game in self.games).items():
          print(f"{color} won {win_count} games ({win_count / game_count * 100}%)")
        return True

      self.start_new_game()

    return False
