import time

from autogamen.game.player import Player
from autogamen.game.types import TurnAction

class BozoPlayer(Player):
  def action(self):
    time.sleep(100)
    return [TurnAction.Move, None]

  def accept_doubling_cube(self):
    return False
