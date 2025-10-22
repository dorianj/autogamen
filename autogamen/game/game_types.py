import random
from enum import Enum
from functools import total_ordering


class Color(Enum):
  White = 0
  Black = 1

  def opponent(self) -> "Color":
    if self == Color.White:
      return Color.Black
    else:
      return Color.White


class _Point:
  __slots__ = ['count', 'color']
  def __init__(self, count: int = 0, color: "Color | None" = None) -> None:
    self.count = count
    self.color = color

  def validate(self) -> None:
    # Not called by constructor for better performance
    if self.count != 0 and self.color is None:
      raise Exception("Color must be set for nonzero pip count.")

    if self.count == 0 and self.color is not None:
      raise Exception("Color can't be set for zero pip count.")

  def __str__(self) -> str:
    if self.color is None:
      return "<Empty Point>"
    else:
      return f"<Point {self.color}/{self.count}>"

  def __repr__(self) -> str:
    return self.__str__()

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _Point):
      return NotImplemented
    return self.count == other.count and self.color == other.color

  def mutable_copy(self) -> "Point":
    return Point(self.count, self.color)

  def frozen_copy(self) -> "FrozenPoint":
    if isinstance(self, FrozenPoint):
      return self
    else:
      return FrozenPoint(self.count, self.color)

  def is_empty(self) -> bool:
    return self.color is None

  def can_hit(self, color: Color) -> bool:
    return (not self.is_empty()) and (self.color != color) and (self.count == 1)

  def can_land(self, color: Color) -> bool:
    """Is :color: is allowed to add here or hit here?"""
    return self.is_empty() or self.can_hit(color) or (self.color == color)


class FrozenPoint(_Point):
  def __hash__(self) -> int:
    if self.count == 0:
      return 0
    else:
      assert self.color is not None
      return self.color.value << 10 | self.count


class Point(_Point):
  def add(self, color: Color) -> None:
    if self.color is None:
      self.color = color

    if self.color != color:
      raise Exception("Point.add on an opponent-occupied point - should .hit instead")

    self.count += 1

  def hit(self, color: Color) -> None:
    if self.color == color:
      raise Exception("Point.hit on own point -- should .add instead")

    if self.count != 1:
      raise Exception("Point.hit on a blocked point (logic error)")

    self.color = color

  def subtract(self, color: Color) -> None:
    if self.count == 0:
      raise Exception("Point.subtract on a zero point")

    self.count -= 1

    if self.count == 0:
      self.color = None


class TurnAction(Enum):
  Move = 1
  DoublingCube = 2
  Pass = 3


@total_ordering
class Move:
  # Bar's point number is 0, which is special. Board relies on this behavior
  # in possible_moves to avoid code duplication.
  Bar = 0

  def __init__(self, color: Color, point_number: int, distance: int) -> None:
    self.point_number = point_number
    self.distance = distance
    self.color = color

    self.destination_is_off = self._destination_is_off()
    self.destination_point_number: int | None = None if self.destination_is_off else self._destination_point_number()

  def __str__(self) -> str:
    source = "bar" if self.point_number is Move.Bar else self.point_number
    destination = "bar" if self.destination_is_off else self.destination_point_number
    return f"<{source}+{self.distance} -> {destination}>"

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Move):
      return NotImplemented
    return (
      self.point_number == other.point_number and
      self.distance == other.distance and
      self.color == other.color
    )

  def __lt__(self, other: object) -> bool:
    if not isinstance(other, Move):
      return NotImplemented
    return (
      (self.point_number, self.distance, self.color) <
      (other.point_number, other.distance, other.color)
    )

  def __hash__(self) -> int:
    return hash((self.point_number, self.distance, self.color))

  def __repr__(self) -> str:
    return self.__str__()

  def _destination_point_number(self) -> int:
    direction = -1 if self.color == Color.Black else 1

    if self.point_number == Move.Bar:
      effective_start = 25 if self.color == Color.Black else 0
    else:
      effective_start = self.point_number

    return effective_start + self.distance * direction

  def _destination_is_off(self) -> bool:
    return self._destination_point_number() > 24 or self._destination_point_number() <= 0


class Dice:
  def __init__(self, count: int = 2, roll: tuple[int, int] | None = None) -> None:
    self.count = count
    if roll is None:
      self.roll = (random.randint(1, 6), random.randint(1, 6))
    else:
      self.roll = roll

  def effective_roll(self) -> tuple[int, ...]:
    """Returns a list of rolls; for doubles, they will be duplicated.
    """
    if self.roll[0] == self.roll[1]:
      return (self.roll[0], self.roll[0], self.roll[0], self.roll[0])
    else:
      return self.roll
