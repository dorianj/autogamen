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

  input_neurons = 190
  hidden_neurons = 20

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(self.input_neurons, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(28, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(self.hidden_neurons, 4)
    )

    with torch.no_grad():
      print(f"layers@0: {self.layers[0]}")


  def vectorize_board(self, board):
    # TODO: change this
    board_state = [point.count * (1 if point.color == Color.White else -1)
                   for point in board.points]
    for color in Color:
      board_state.append(board.off[color])
      board_state.append(board.bar[color])

    return torch.FloatTensor(board_state)

  def forward(self, x):
    return self.layers(x)

  def update_weights(self, p, p_next):
    pass


class MLPPlayer(Player):
  """Picks the best move in the universe every time.
  """
  def __init__(self, color, net, learning=False):
    super().__init__(color)
    self.net = net
    self.learning = learning

  def p(self, board):
    return self.net(self.net.vectorize_board(board))

  def score(self, p):
    [white_wins, black_wins, white_gammons, black_gammons] = p
    if self.color == Color.White:
      return white_wins * 1 + white_gammons * 2
    else:
      return black_wins * 1 + black_gammons * 2

  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    if self.learning:
      p = self.p(self.game.board)

    possible_boards = set((self.game.board.clone_apply_moves(moves).frozen_copy(), moves)
                          for moves in possible_moves)

    best_score = -10000
    best_p = None
    best_moves = None
    for (board, moves) in possible_boards:
      p = self.p(board)
      score = self.score(p)
      if score > best_score:
        best_score = score
        best_moves = moves
        best_p = p

    if self.learning:
      self.net.update_weights(p, p_next)

    return [TurnAction.Move, best_moves]

  def accept_doubling_cube(self):
    return False

  def end_game(self, game):
    super().end_game(game)

    if self.learning:
      """TODO"""

