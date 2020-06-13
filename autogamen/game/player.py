class Player:
  def __init__(self, color):
    self.color = color
    self.game = None

  def start_game(self, game):
    self.game = game

  def action(self, possible_moves):
    """Called at the start of a turn.
    Return: [TurnAction, moves?]
    """
    raise Exception("action not implemented")

  def accept_doubling_cube(self):
    """Called when opponent has offered the doubling cube.
    Return: boolean, true to accept; false to forfeit
    """
    raise Exception("doubling_cube not implemented")

  def end_game(self, game):
    pass
    self.game = None


  def possible_moves(self, game):
    pass
