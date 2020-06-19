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

    return [TurnAction.Move, random.choice(sorted(possible_moves))[0]]

  def accept_doubling_cube(self):
    return False



class RunningPlayer(Player):
  """Picks the move that minimizes own pip count.
  """
  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    boards_by_pip_count = defaultdict(set)
    for moves, board in possible_moves:
      boards_by_pip_count[board.pip_count()[self.color]].add(moves)

    possible_moves = boards_by_pip_count[min(boards_by_pip_count.keys())]
    return [TurnAction.Move, random.choice(sorted(possible_moves))]

  def accept_doubling_cube(self):
    return True


class DeltaPlayer(Player):
  """Picks the move that maximizes pip count delta.
  """
  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    boards_by_pip_delta = defaultdict(set)
    for moves, board in possible_moves:
      pip_delta = board.pip_count()[self.color.opponent()] - board.pip_count()[self.color]
      boards_by_pip_delta[pip_delta].add(moves)

    possible_moves = boards_by_pip_delta[max(boards_by_pip_delta.keys())]
    return [TurnAction.Move, random.choice(sorted(possible_moves))]

  def accept_doubling_cube(self):
    return True
