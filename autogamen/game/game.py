import random
import time

from .types import Color, Point, TurnAction
from .board import Board

class Dice:
  def __init__(self, count=2):
    self.count = count
    self.roll = (random.randint(1,6), random.randint(1, 6))

  def effective_roll(self):
    """Returns a list of rolls; for doubles, they will be duplicated.
    """
    if self.roll[0] == self.roll[1]:
      return [self.roll[0], self.roll[0], self.roll[0], self.roll[0]]
    else:
      return self.roll


class Game:
  def __init__(self, players):
    self.board = Board([
      Point(2, Color.White),
      Point(0),
      Point(0),
      Point(0),
      Point(0),
      Point(5, Color.Black),
      Point(0),
      Point(3, Color.Black),
      Point(0),
      Point(0),
      Point(0),
      Point(5, Color.White),
      Point(5, Color.Black),
      Point(0),
      Point(0),
      Point(0),
      Point(3, Color.White),
      Point(0),
      Point(5, Color.White),
      Point(0),
      Point(0),
      Point(0),
      Point(0),
      Point(2, Color.Black),
    ])

    self.active_player_idx = None
    self.active_dice = [0, 0]

    self.doubling_owner = None
    self.doubling_cube = 1

    self.players = players # white,black

    self.turn_number = 0

    self.winner = None
    self.outcome = None

  def roll_starting(self):
    while True:
      dice = Dice()
      [white_die, black_die] = dice.roll
      if white_die != black_die:
        return ((white_die > black_die), dice)

  def start(self):
    for player in self.players:
      player.start_game(self)

    [white_starts, self.active_dice] = self.roll_starting()
    self.active_player_idx = 0 if white_starts else 1

    print(f"First roll: {self.active_dice.roll}; starting player: {self.active_color()}")

  def end(self, winner, outcome):
    self.winner = winner
    self.outcome = outcome

    # Inform the players of their fate
    for player in self.players:
      player.end_game(self)

  def run_turn(self):
    """Runs a single turn. Returns true if the game is still in-progress.
    """
    self.turn_number += 1
    print(f"Turn {self.turn_number}: {self.active_color()} has roll {self.active_dice.roll}:")

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
    return self.players[self.active_player_idx]

  def inactive_player(self):
    return self.players[(self.active_player_idx + 1) % 2]

  def white_is_active(self):
    return self.active_player == 0

  def active_color(self):
    return Color.White if self.active_player == 0 else Color.Black
