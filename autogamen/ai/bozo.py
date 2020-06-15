import time
import random

from autogamen.game.player import Player
from autogamen.game.types import TurnAction

class BozoPlayer(Player):
  """This player picks a random move every time, and never accepts
     the doubling cube.
  """
  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    return [TurnAction.Move, random.choice(list(possible_moves))]

  def accept_doubling_cube(self):
    return False
