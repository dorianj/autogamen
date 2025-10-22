from collections import defaultdict
from typing import TYPE_CHECKING

import torch
import torch.autograd
import torch.nn as nn

from autogamen.game.game_types import Color, TurnAction
from autogamen.game.player import Player

if TYPE_CHECKING:
    from autogamen.game.board import _Board
    from autogamen.game.game import Game


class Net(nn.Module):
  input_neurons = 198
  hidden_neurons = 50
  learning_rate = 0.1

  def __init__(self) -> None:
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(self.input_neurons, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(self.hidden_neurons, self.hidden_neurons),
      nn.Sigmoid(),
      nn.Linear(self.hidden_neurons, 1)
    )

    self.lambda_ = 0.5
    self.eligibility_traces = [
      torch.zeros(weights.shape, requires_grad=False)
      for weights in list(self.parameters())
    ]

    for param in self.parameters():
      nn.init.zeros_(param)

  def vectorize_board(self, board: "_Board", active_color: Color) -> torch.Tensor:
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

    out.append(board.off[Color.White.value])
    out.append(board.off[Color.Black.value])

    out.append(board.bar[Color.White.value])
    out.append(board.bar[Color.Black.value])

    if len(out) != self.input_neurons:
      raise Exception("vectorize_board is broken")

    return torch.FloatTensor(out)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = self.layers(x)
    return result

  def reset_eligibility_traces(self) -> None:
    """Reset eligibility traces to zero (called at start of each game)"""
    for i in range(len(self.eligibility_traces)):
      self.eligibility_traces[i].zero_()

  def update_weights(self, p: torch.Tensor, p_next: torch.Tensor) -> torch.Tensor:
    self.zero_grad()

    p.backward()

    with torch.no_grad():
      td_error = p_next - p

      params = list(self.parameters())

      for i, weights in enumerate(params):
        grad = weights.grad if weights.grad is not None else torch.zeros_like(weights)
        self.eligibility_traces[i] = self.lambda_ * self.eligibility_traces[i] + grad

        # w <- w + alpha * td_error * z
        new_weights = weights + self.learning_rate * td_error * self.eligibility_traces[i]
        weights.copy_(new_weights)

    return td_error


class MLPPlayer(Player):
  """Picks the best move in the universe every time.
  """
  def __init__(self, color: Color, net: Net, learning: bool = False) -> None:
    super().__init__(color)
    self.net = net
    self.learning = learning

  def start_game(self, game: "Game") -> None:
    super().start_game(game)
    if self.learning:
      self.net.reset_eligibility_traces()

  def p(self, board: "_Board") -> torch.Tensor:
    result: torch.Tensor = self.net(self.net.vectorize_board(board, self.color))
    return result

  def win_probability(self, p: torch.Tensor) -> float:
    [white_wins] = p
    win_prob: float
    if self.color == Color.White:
      win_prob = white_wins.item()
    else:
      win_prob = 1 - white_wins.item()
    return win_prob

  def action(self, possible_moves: set[tuple[tuple, "_Board"]]) -> list:
    if not len(possible_moves):
      return [TurnAction.Pass]

    if self.learning:
      assert self.game is not None
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

    best_p, best_board, best_move = max(
      scored_boards,
      key=lambda i: self.win_probability(i[0])
    )

    if self.learning:
      if best_board.winner() is not None:
        # Terminal state: z = 1 for White win, 0 for Black win
        reward = torch.tensor(1.0 if self.color == Color.White else 0.0)
      else:
        # Non-terminal state: next predicted value (detach to avoid backprop)
        reward = best_p.detach()

      self.net.update_weights(p, reward)

    return [TurnAction.Move, best_move]

  def accept_doubling_cube(self) -> bool:
    return False
