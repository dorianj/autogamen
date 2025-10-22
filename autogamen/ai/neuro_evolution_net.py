"""neural network for neuro-evolution training - no TD machinery, just forward pass + genetic ops"""
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from autogamen.ai.players import Player
from autogamen.game.game_types import Color, TurnAction

if TYPE_CHECKING:
    from autogamen.game.board import _Board


class NENet(nn.Module):
    """net for neuro-evolution - stripped down to just forward pass + genetic operators"""
    input_neurons = 198
    hidden_neurons = 50

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.input_neurons, self.hidden_neurons),
            nn.Sigmoid(),
            nn.Linear(self.hidden_neurons, self.hidden_neurons),
            nn.Sigmoid(),
            nn.Linear(self.hidden_neurons, 1)
        )

        for param in self.parameters():
            nn.init.zeros_(param)

    @property
    def weights(self) -> list[torch.Tensor]:
        """weights as list for pickle serialization (legacy neuro-evolution format)"""
        return [p.data.clone() for p in self.parameters()]

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

    def mutate(self, rate: float) -> "NENet":
        """gaussian mutation of weights"""
        mutated = NENet()
        with torch.no_grad():
            for orig_param, mut_param in zip(self.parameters(), mutated.parameters(), strict=True):
                noise = torch.randn_like(orig_param) * rate
                mut_param.copy_(orig_param + noise)
        return mutated

    def breed(self, other: "NENet") -> "NENet":
        """uniform crossover - randomly pick weights from each parent"""
        child = NENet()
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(
                child.parameters(), self.parameters(), other.parameters(), strict=True
            ):
                mask = torch.rand_like(p1_param) > 0.5
                child_param.copy_(torch.where(mask, p1_param, p2_param))
        return child


class NEMLPPlayer(Player):
    """player using NENet for neuro-evolution training"""
    def __init__(self, color: Color, net: NENet) -> None:
        super().__init__(color)
        self.net = net

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

        moves_by_board = defaultdict(list)
        for move, board in possible_moves:
            moves_by_board[board].append(move)

        scored_boards = [
            (self.p(board), board, moves[0])
            for board, moves in moves_by_board.items()
        ]

        best_p, best_board, best_move = max(
            scored_boards,
            key=lambda i: self.win_probability(i[0])
        )

        return [TurnAction.Move, best_move]

    def accept_doubling_cube(self) -> bool:
        return False
