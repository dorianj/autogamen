from collections import defaultdict
import random

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

from autogamen.game.player import Player
from autogamen.game.types import Color, TurnAction


class Net(nn.Module):
  input_neurons = 198
  hidden_neurons = 50
  learning_rate = 0.1

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(self.input_neurons, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(self.hidden_neurons, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(self.hidden_neurons, 1)
    )

    torch.autograd.set_detect_anomaly(True)
    self.lambda_ = 0.5
    self.eligibility_traces = [
      torch.zeros(weights.shape, requires_grad=False)
      for weights in list(self.parameters())
    ]

    for param in self.parameters():
      nn.init.zeros_(param)

  def vectorize_board(self, board, active_color):
    out = []
    for point in board.points:
      val = [
        1 if point.count >= 1 else 0,
        1 if point.count >= 2 else 0,
        1 if point.count >= 3 else 0,
        (point.count - 3) / 2,
      ]
      empty = [0, 0, 0, 0]

      if point.color == Color.White:
        out.extend(val)
        out.extend(empty)
      elif point.color == Color.Black:
        out.extend(empty)
        out.extend(val)
      else:
        out.extend(empty)
        out.extend(empty)

    out.append(1 if active_color == Color.White else 0)
    out.append(1 if active_color == Color.Black else 0)

    out.append(board.off[Color.White])
    out.append(board.off[Color.Black])

    out.append(board.bar[Color.White])
    out.append(board.bar[Color.Black])

    if len(out) != self.input_neurons:
      raise Exception("vectorize_board is broken")

    return torch.FloatTensor(out)

  def forward(self, x):
    return self.layers(x)

  def update_weights(self, p, p_next):
    print(p, p_next)
    self.zero_grad()

    p.backward()

    with torch.no_grad():
      td_error = p_next - p

      params = list(self.parameters())

      for i, weights in enumerate(params):
        self.eligibility_traces[i] = self.lambda_ * self.eligibility_traces[i] + weights.grad

        # w <- w + alpha * td_error * z
        new_weights = weights + self.learning_rate * td_error * self.eligibility_traces[i]
        weights.copy_(new_weights)

    return td_error


class MLPPlayer(Player):
  """Picks the best move in the universe every time.
  """
  def __init__(self, color, net, learning=False):
    super().__init__(color)
    self.net = net
    self.learning = learning

  def p(self, board):
    return self.net(self.net.vectorize_board(board, self.color))

  def score(self, p):
    [white_wins] = p
    if self.color == Color.White:
      return white_wins
    else:
      return 1 - white_wins

  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    if self.learning:
      p = self.p(self.game.board)

    # Dedupe moves into materialized boards: if multiple moves result in the same
    # board, we needn't score the boards multiple times
    moves_by_board = defaultdict(list)
    for move, board in possible_moves:
      moves_by_board[board].append(move)

    # Score the boards by the neural net
    scored_boards = [
      (self.p(board), board, moves[0])
      for board, moves in moves_by_board.items()
    ]

    # argmax according to score
    sorted_boards = sorted(
      scored_boards,
      key=lambda i: self.score(i[0]),
      reverse=self.color == Color.Black
    )

    best_p, best_board, best_move = sorted_boards[0]

    if self.learning:
      reward = None
      if best_board.winner() is not None:
        reward = 1 if self.color == Color.White else 0
      else:
        reward = best_p

      self.net.update_weights(p, reward)

    return [TurnAction.Move, best_move]

  def accept_doubling_cube(self):
    return False
