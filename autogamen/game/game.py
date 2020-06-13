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
      Point(0),
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
      Point(0),
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
    self.outcome = None

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

  def end(self, winner, outcome):
    self.winner = winner
    self.outcome = outcome

    # Inform the players of their fate
    for player in self.players.values():
      player.end_game(self)

  def run_turn(self):
    """Runs a single turn. Returns true if the game is still in-progress.
    """
    self.turn_number += 1
    print(f"Turn {self.turn_number}: {self.active_color} has roll {self.active_dice.roll}:")

    [turn_action, turn_detail] = self.active_player().action()

    if turn_action is TurnAction.Move:
      print(f"\tmove")
      # TODO: validate the move
      raise Exception("invalid move")
      return True
    elif turn_action is TurnAction.DoublingCube:
      print(f"\tOffers the doubling cube")
      raise Exception("Doubling cube is unimplemented")
      return True
    else:
      raise Exception(f"Unknown turn action {turn_action}")

    time.sleep(10000)

  def active_player(self):
    return self.players[self.active_color]
