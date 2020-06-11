import random

from .types import Point, Color

class Dice:
  def __init__(self, count=2):
    self.count = count
    self.roll = (random.randint(1,6), random.randint(1, 6))


starting_points = [
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
  Point(3, Color.Black),
  Point(0),
  Point(5, Color.White),
  Point(0),
  Point(0),
  Point(0),
  Point(0),
  Point(2, Color.Black),
]

class Game:
  def __init__(self, player_black, player_white):
    self.points = [
      Point(2, Color.White),
      Point(0),
      Point(0),
      Point(0),
      Point(0),
      Point(7, Color.Black),
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
      Point(3, Color.Black),
      Point(0),
      Point(6, Color.White),
      Point(0),
      Point(0),
      Point(0),
      Point(0),
      Point(2, Color.Black),
    ]
    self.active_player = None
    self.active_dice = [0, 0]

    self.doubling_owner = None
    self.doubling_cube = 1

  def roll_starting(self):
    while True:
      dice = Dice()
      [white_die, black_die] = dice.roll
      if white_die != black_die:
        return ((white_die > black_die), dice)

  def start(self):
    [white_starts, self.active_dice] = self.roll_starting()
    self.active_player = Color.White if white_starts else Color.Black

    print(f"Active dice: {self.active_dice}; starting player: {self.active_player}")

  def white_is_active(self):
    return self.active_player == Color.White

"""
need an interface to ask the player what move they'd
like to make, and wait on it. The UI is 1 or 2 players,
can also make a dummy player
"""


