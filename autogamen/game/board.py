from collections import defaultdict

class Board:
  def __init__(self, points):
    if len(points) != 24:
      raise Exception(f"{len(points)} board passed in.")

    self.points = points

    counts = defaultdict(int)
    for point in self.points:
      if point.color is not None:
        counts[point.color] += point.count

    print(counts)
    if list(counts.values()) != [15, 15]:
      raise Exception(f"Invalid number of pips: {counts.values()}")


  def can_bear_off(self, color):
    return False

  def possible_moves(self, color):
    """Returns the list of possible moves. Each item is a Set of Moves
    """
    rolls = self.active_dice.effective_roll()

    # First, check to see if player is eligible to bear off.
