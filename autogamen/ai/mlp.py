from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
import torch.autograd
import torch.nn as nn

from autogamen.ai.players import Player
from autogamen.game import vector_game as vg
from autogamen.game.game_types import Color, TurnAction

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

  def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
    """batched forward pass for multiple board states

    x: shape (N, 198) - batch of board vectors
    returns: shape (N, 1) - win probability predictions
    """
    return self.forward(x)

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


class VectorMLPPlayer:
  """vector-native MLP player for high-performance training.

  operates directly on numpy vectors from vector_game, uses batched
  neural net evaluation for speedup.
  """
  def __init__(self, color: Color, net: Net, learning: bool = False) -> None:
    self.color = color
    self.net = net
    self.learning = learning
    self.prev_vec: npt.NDArray[np.float32] | None = None
    self.prev_p: torch.Tensor | None = None

  def reset_eligibility_traces(self) -> None:
    """called at start of each game"""
    if self.learning:
      self.net.reset_eligibility_traces()
      self.prev_vec = None
      self.prev_p = None

  def p_batch(self, vecs: npt.NDArray[np.float32]) -> torch.Tensor:
    """evaluate batch of board vectors.

    vecs: shape (N, 198) numpy array
    returns: shape (N, 1) tensor of win probabilities
    """
    batch = torch.from_numpy(vecs)
    return self.net.forward_batch(batch)

  def p_single(self, vec: npt.NDArray[np.float32]) -> torch.Tensor:
    """evaluate single board vector.

    vec: shape (198,) numpy array
    returns: shape (1,) tensor
    """
    tensor = torch.from_numpy(vec).unsqueeze(0)  # shape: (1, 198)
    result = self.net.forward(tensor)  # shape: (1, 1)
    return result.squeeze(0)  # shape: (1,)

  def win_probability(self, p: torch.Tensor) -> float:
    """convert net output to win probability for this player"""
    white_wins = p.item()
    if self.color == Color.White:
      return white_wins
    else:
      return 1 - white_wins

  def choose_move(
    self,
    vec: npt.NDArray[np.float32],
    color: Color,
    dice: tuple[int, ...],
  ) -> tuple[tuple, npt.NDArray[np.float32]]:
    """choose best move using batched neural net evaluation.

    returns: (best_move_tuple, resulting_board_vector)
    """
    possible_moves = vg.possible_moves(vec, color, dice)

    if not possible_moves:
      return ((), vec)

    if self.learning:
      self.prev_vec = vec.copy()
      self.prev_p = self.p_single(vec)

    # stack all result vectors for batch evaluation
    result_vecs = np.stack([result_vec for _, result_vec in possible_moves])  # shape: (N, 198)

    # single batched forward pass
    scores = self.p_batch(result_vecs)  # shape: (N, 1)

    # find best move for this player
    win_probs = torch.tensor([self.win_probability(scores[i]) for i in range(len(scores))])
    best_idx = int(win_probs.argmax().item())

    best_move, best_vec = possible_moves[best_idx]
    best_p = scores[best_idx]

    # TD-learning weight update
    if self.learning:
      winner = vg.winner(best_vec)

      if winner is not None:
        # terminal state: reward is 1 for white win, 0 for black win
        reward = torch.tensor([1.0 if winner == Color.White else 0.0])
      else:
        # non-terminal: use next predicted value (detached)
        reward = best_p.detach()

      assert self.prev_p is not None
      self.net.update_weights(self.prev_p, reward)

    return best_move, best_vec
