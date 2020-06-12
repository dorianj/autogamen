from enum import Enum


class Color(Enum):
  White = 1
  Black = 2


class Point:
  def __init__(self, count, color=None):
    if count != 0 and color is None:
      raise Exception("Color must be set for nonzero pip count.")

    self.count = count
    self.color = color


class TurnAction(Enum):
  Move = 1
  DoublingCube = 2


class Move:
  def __init__(self, point_number, distance):
    self.point_number = point_number
    self.distance = distance

