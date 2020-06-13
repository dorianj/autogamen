import random
import time

from .types import Color, Point, TurnAction, Dice
from .board import Board


class Game:
  def __init__(self, players):
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

    if len(players) != 2 or list(self.players.keys()) != list(Color):
      raise Exception("Game constructor: Invalid players provided")

    # Turn tracking, will by initialized by start() and mutated by run_turn()
    self.active_color = None
    self.active_dice = None
    self.doubling_owner = None
    self.doubling_cube = 1

    # Internal tracking, no implications on gameplay. Mutated by run_turn()
    self.turn_number = 0

    # Set by end()
    self.winner = None
    self.points = 0

  def roll_starting(self):
    while True:
      dice = Dice()
      [white_die, black_die] = dice.roll
      if dice.roll[0] != dice.roll[1]:
        return dice

  def start(self):
    for player in self.players.values():
      player.start_game(self)

    self.active_dice = self.roll_starting()
    self.active_color = Color.White if self.active_dice.roll[0] > self.active_dice.roll[1] else Color.Black

    print(f"First roll: {self.active_dice.roll}; starting player: {self.active_color}")

  def end(self, winner, points):
    self.winner = winner
    self.points = points

    # Inform the players of their fate
    for player in self.players.values():
      player.end_game(self)

  def check_winning_condition(self):
    if self.board.winner():
      self.end(self.players[self.board.winner()], self.board.winner_stakes() * self.doubling_cube)
      return True
    else:
      return False

  def run_turn(self):
    """Runs a single turn.
    """
    self.active_color = self.active_color.opponent()
    self.active_dice = Dice()
    self.turn_number += 1
    print(f"Turn {self.turn_number}: {self.active_color} has roll {self.active_dice.roll}:")

    possible_moves = self.board.possible_moves(self.active_color, self.active_dice)
    turn_result = self.active_player().action(set(possible_moves))
    turn_action = turn_result[0]

    if turn_action is TurnAction.Move:
      moves = turn_result[1]
      print(f"\tMoves: {moves}")

      if moves not in possible_moves:
        raise Exception("Illegal move attempted!")

      [self.board.apply_move(move) for move in moves]
    elif turn_action is TurnAction.DoublingCube:
      print(f"\tOffers the doubling cube")
      raise Exception("Doubling cube is unimplemented")
    elif turn_action is TurnAction.Pass:
      if len(possible_moves):
        raise Exception("Illegal pass attempted!")
      print("\tNo play available, player passes")
    else:
      raise Exception(f"Unknown turn action {turn_action}")

    self.check_winning_condition()


  def active_player(self):
    return self.players[self.active_color]
