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
