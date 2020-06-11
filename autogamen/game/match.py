from .game import Game

class Match:
  def __init__(self):
    self.players = [None, None]
    self.current_game = Game(*self.players)

  def start(self):
    self.current_game.start()
