from collections import defaultdict
import random

from autogamen.game.player import Player
from autogamen.game.types import TurnAction


class BozoPlayer(Player):
  """Picks a random move every time.
  """
  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    return [TurnAction.Move, random.choice(sorted(possible_moves))]

  def accept_doubling_cube(self):
    return False



class RunningPlayer(Player):
  """This player picks the move that minimizes own pip count.
  """
  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    boards_by_pip_count = defaultdict(set)
    for moves in possible_moves:
      board = self.game.board.clone_apply_moves(moves)
      boards_by_pip_count[board.pip_count()[self.color]].add(moves)

    possible_moves = boards_by_pip_count[min(boards_by_pip_count.keys())]
    return [TurnAction.Move, random.choice(sorted(possible_moves))]

  def accept_doubling_cube(self):
    return True
