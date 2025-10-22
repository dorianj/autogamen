import logging
from typing import Any

from .board import Board
from .game_types import Color, Dice, Point, TurnAction


class Game:
  def __init__(self, players: Any) -> None:
    self.board = Board([
      Point(2, Color.White),
      Point(),
      Point(),
      Point(),
      Point(),
      Point(5, Color.Black),
      Point(),
      Point(3, Color.Black),
      Point(),
      Point(),
      Point(),
      Point(5, Color.White),
      Point(5, Color.Black),
      Point(),
      Point(),
      Point(),
      Point(3, Color.White),
      Point(),
      Point(5, Color.White),
      Point(),
      Point(),
      Point(),
      Point(),
      Point(2, Color.Black),
    ])

    # Player state tracking. Immutable.
    self.players = {player.color: player for player in players}

    if len(players) != 2 or set(self.players.keys()) != set(Color):
      raise Exception(f"Game constructor: Invalid players provided: {self.players}")

    # Turn tracking, will by initialized by start() and mutated by run_turn()
    self.active_color: Color | None = None
    self.active_dice: Dice | None = None
    self.doubling_owner: Color | None = None
    self.doubling_cube = 1

    # Internal tracking, no implications on gameplay. Mutated by run_turn()
    self.turn_number = 0

    # Set by end()
    self.winner = None
    self.points = 0

  def roll_starting(self) -> Dice:
    while True:
      dice = Dice()
      [white_die, black_die] = dice.roll
      if dice.roll[0] != dice.roll[1]:
        return dice

  def start(self) -> None:
    for player in self.players.values():
      player.start_game(self)

    self.active_dice = self.roll_starting()
    self.active_color = Color.White if self.active_dice.roll[0] > self.active_dice.roll[1] else Color.Black

    logging.debug(f"First roll: {self.active_dice.roll}; starting player: {self.active_color}")

  def end(self, winner: Any, points: int) -> None:
    self.winner = winner
    self.points = points

    # Inform the players of their fate
    for player in self.players.values():
      player.end_game(self)

  def check_winning_condition(self) -> bool:
    if self.board.winner():
      self.end(self.players[self.board.winner()], self.board.winner_stakes() * self.doubling_cube)
      return True
    else:
      return False

  def pre_turn(self) -> None:
    """Set up the board for the upcoming turn without blocking
    """
    assert self.active_color is not None
    self.active_color = self.active_color.opponent()
    self.active_dice = Dice()
    self.turn_number += 1
    logging.debug(f"Turn {self.turn_number}: {self.active_color} has roll {self.active_dice.roll}:")

  def turn_blocking(self) -> None:
    """Run the turn, blocking until player chooses an action
    """
    assert self.active_color is not None
    assert self.active_dice is not None
    possible_moves = self.board.possible_moves(self.active_color, self.active_dice)
    turn_result = self.active_player().action(possible_moves)
    turn_action = turn_result[0]

    if turn_action is TurnAction.Move:
      move = turn_result[1]
      logging.debug(f"\tMoved: {move}")

      if not any(move == m[0] for m in possible_moves):
        raise Exception("Illegal move attempted!")

      for m in move:
        self.board.apply_move(m)
    elif turn_action is TurnAction.DoublingCube:
      logging.debug("\tOffers the doubling cube")
      raise Exception("Doubling cube is unimplemented")
    elif turn_action is TurnAction.Pass:
      if len(possible_moves):
        raise Exception("Illegal pass attempted!")
      logging.debug("\tNo play available, player passes")
    else:
      raise Exception(f"Unknown turn action {turn_action}")

    self.check_winning_condition()

  def run_turn(self) -> None:
    """Runs a single turn, blocks
    """
    self.pre_turn()
    self.turn_blocking()

  def active_player(self) -> Any:
    assert self.active_color is not None
    return self.players[self.active_color]
