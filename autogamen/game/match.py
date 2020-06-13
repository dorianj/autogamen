from .game import Game

class Match:
  def __init__(self, players):
    self.players = players

  def start(self):
    self.start_new_game()

  def start_new_game(self):
    self.current_game = Game(self.players)
    self.current_game.start()

  def tick(self):
    self.current_game.run_turn()

    if self.current_game.winner:
      print(f"Game ended! {self.current_game.winner} won with {self.current_game.points} points")
      self.start_new_game()
