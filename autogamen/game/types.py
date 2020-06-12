from enum import Enum
import random


class Color(Enum):
  White = 1
  Black = 2

  def opposite(self):
    if self == Color.White:
      return Color.Black
    else:
      return Color.White


class Point:
  def __init__(self, count=0, color=None):
    if count != 0 and color is None:
      raise Exception("Color must be set for nonzero pip count.")

    if count == 0 and color is not None:
      raise Exception("Color can't be set for zero pip count.")

    self.count = count
    self.color = color

  def __str__(self):
    if self.color is None:
      return "(Empty Point)"
    else:
      return f"(Point {self.color}/{self.count})"

  def __eq__(self, other):
    return self.count == other.count and self.color == other.color

  def is_empty(self):
    return self.color is None

  def can_hit(self, color):
    return (not self.is_empty()) and (self.color != color) and (self.count == 1)

  def can_land(self, color):
    """Returns whether it's allowed to add here or hit here."""
    return self.is_empty() or self.can_hit(color) or (self.color == color)

  def add(self, color):
    if self.color is None:
      self.color = color

    if self.color != color:
      raise Exception("Point.add on an opponent-occupied point - should .hit instead")

    self.count += 1

  def hit(self, color):
    if self.color == color:
      raise Exception("Point.hit on own point -- should .add instead")

    if self.count != 1:
      raise Exception("Point.hit on a blocked point (logic error)")

    self.color = color

  def subtract(self, color):
    if self.count == 0:
      raise Exception("Point.subtract on a zero point")

    self.count -= 1

    if self.count == 0:
      self.color = None


class TurnAction(Enum):
  Move = 1
  DoublingCube = 2


class Move:
  # Bar's point number is 0, which is special. Board relies on this behavior
  # in possible_moves to avoid code duplication.
  Bar = 0


  def __init__(self, color, point_number, distance):
    self.point_number = point_number
    self.distance = distance
    self.color = color

  def __str__(self):
    source = "bar" if self.point_number is Move.Bar else self.point_number
    destination = "bar" if self.destination_is_off() else self._destination_point_number()
    return f"({source}+{self.distance} -> {destination})"

  def __eq__(self, other):
    return (
      self.point_number == other.point_number and
      self.distance == other.distance and
      self.color == other.color
    )

  def __hash__(self):
    return hash((self.point_number, self.distance, self.color))

  def __repr__(self):
    return self.__str__()

  def _destination_point_number(self):
    direction = -1 if self.color == Color.Black else 1

    if self.point_number == Move.Bar:
      effective_start = 25 if self.color == Color.Black else 0
    else:
      effective_start = self.point_number

    return effective_start + self.distance * direction

  def destination_is_off(self):
    return self._destination_point_number() > 24 or self._destination_point_number() <= 0

  def destination_point_number(self):
    if self.destination_is_off():
      raise Exception("Move: Can't get destination point if into bar")

    return self._destination_point_number()

class Dice:
  def __init__(self, count=2, roll=None):
    self.count = count
    self.roll = roll or (random.randint(1,6), random.randint(1, 6))

  def effective_roll(self):
    """Returns a list of rolls; for doubles, they will be duplicated.
    """
    if self.roll[0] == self.roll[1]:
      return [self.roll[0], self.roll[0], self.roll[0], self.roll[0]]
    else:
      return self.roll
