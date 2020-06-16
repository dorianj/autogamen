from collections import defaultdict
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from autogamen.game.player import Player
from autogamen.game.types import Color, TurnAction


class Net(nn.Module):
  """
    Inputs:
      * 24: One per point, negative for black and white for white
      * 2: One per player, count on bar
      * 2: One per player, count beared off
    Outputs: 4, probability white or black wins or with a gammon
  """

  hidden_neurons = 28

  def __init__(self, weights):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(28, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(self.hidden_neurons, 4)
    )
    self.weights = weights # TODO: apply weights

  def vectorize_board(self, board):
    board_state = [point.count * (1 if point.color == Color.White else -1)
                   for point in board.points]
    for color in Color:
      board_state.append(board.off[color])
      board_state.append(board.bar[color])

    return torch.FloatTensor(board_state)

  def forward(self, x):
    return self.layers(x)

  def breed(self, other):
    """Combine weights with :other: Net, returns a new Net
    """
    return Net(list(self.weights))

  def mutate(self, factor):
    """Randomly change weights; returns a new Net
    """
    new_weights = list(self.weights)
    return Net(new_weights)


  @classmethod
  def random_net(cls):
    return Net([
      torch.randn(28),
      torch.randn(cls.hidden_neurons),
      torch.randn(4),
    ])


class MLPPlayer(Player):
  """Picks a random move every time.
  """
  def __init__(self, color, net):
    super().__init__(color)
    self.net = net

  def score_board(self, board):
    weights = self.net(self.net.vectorize_board(board))
    [white_wins, black_wins, white_gammons, black_gammons] = weights

    if self.color == Color.White:
      return white_wins * 1 + white_gammons * 2
    else:
      return black_wins * 1 + black_gammons * 2


  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    possible_boards = set((self.game.board.clone_apply_moves(moves).frozen_copy(), moves)
                          for moves in possible_moves)

    best_score = -10000
    best_moves = None
    for (board, moves) in possible_boards:
      score = self.score_board(board)
      if score > best_score:
        best_score = score
        best_moves = moves

    return [TurnAction.Move, best_moves]

  def accept_doubling_cube(self):
    return False

  ##

