from .game import Game

class Match:
  def __init__(self, players):
    self.players = players
    self.current_game = Game(self.players)

  def start(self):
    self.current_game.start()

  def tick(self):
    # TODO: logic to make new game whereupon a game is won
    self.current_game.run_turn()
