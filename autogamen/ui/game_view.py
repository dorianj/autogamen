from collections import namedtuple
from math import floor

from autogamen.game.game_types import Color


# UI geometry types
Coord = namedtuple('Coord', ['x', 'y'])
Rect = namedtuple('Rect', ['width', 'height'])
Area = namedtuple('Area', ['offset', 'rect'])


def choose_color(white, white_color="#ffffff", black_color="#000000"):
  """Ternary statement with defaults."""
  return white_color if white else black_color

class GameView:
  def __init__(self, canvas, area, game):
    self.canvas = canvas
    self.area = area
    self.game = game

    self.bar_width = 40
    self.off_width = 40
    self.playable_width = self.area.rect.width - self.off_width
    self.point_width = (self.playable_width - self.bar_width) / 12.0
    self.point_height = floor(self.area.rect.height / 2 * 0.85)
    # Size pips such that 5 are visible without overlapping
    self.pip_size = floor(min(
      self.point_height / 5,
      self.point_width * 0.95
    ))
    self.doubling_cube_size = 25
    self.die_size = 18

  def offset(self, *args):
    """ Returns the offset point. Accepts x/y args or a single Coord
    """
    point = args[0] if len(args) == 1 else Coord(args[0], args[1])
    return Coord(point.x + self.area.offset.x, point.y + self.area.offset.y)

  def draw_chrome(self):
    """ Draw a border around the board
    """
    self.canvas.create_rectangle(
      self.offset(0, 0),
      self.offset(self.area.rect.width, self.area.rect.height),
    )

  def _point_coord(self, point_number):
    """Returns the Coord coordinate for a given point number
    """
    mod = (point_number - 1) % 12
    x_index = None
    y_pos = None

    if point_number > 12:
      y_pos = 1
      x_index = mod
    else:
      x_index = 12 - mod - 1
      y_pos = self.area.rect.height

    x_pos = self.point_width * x_index

    if x_index >= 6:
      x_pos += self.bar_width

    return Coord(x_pos, y_pos)

  def _y_reflected(self, y, direction):
    """Given a coordinate :y:, position it given :direction:. If direction is
       1 (downwards), y is unadjusted. If direction is -1 (upwards), it will be
       subtracted from the height of the board.
    """
    return y if direction == 1 else self.area.rect.height - y

  def _point_direction(self, point_number):
    """Returns a direction, that is, -1 for upwards and 1 for downwards. Useful
       when drawing stuff going up or down the board."""
    return -1 if point_number <= 12 else 1

  def _color_direction(self, color):
    """Same as _point_direction, but for color's home sides.
    """
    return 1 if color is Color.Black else -1

  def draw_point_board(self, point_number):
    coord = self._point_coord(point_number)
    top_sign = self._point_direction(point_number)

    # Draw the triangle
    self.canvas.create_polygon(
      self.offset(coord.x, coord.y),
      self.offset(coord.x + self.point_width, coord.y),
      self.offset(coord.x + self.point_width / 2, coord.y + self.point_height * top_sign),
      fill=choose_color(point_number % 2, "#990000", "#A5A5A5"),
      tags=[f"point:{point_number}"]
    )

    # Draw the point number indicator
    self.canvas.create_text(
      self.area.offset.x + coord.x + self.point_width / 2,
      self.area.offset.y + coord.y + 10 * top_sign,
      text=point_number,
      tags=[f"point:{point_number}"]
    )

  def draw_pip_stack(self, count, color, direction, area, tags=()):
    """Draw a stack of pips
      :count: int, number of pips to draw
      :color: Color
      :direction: int, 1 for up-to-down, -1 for down-to-up
      :area: Area, bounding box. Coord is top left for downwards, bottom left for upwards
    """
    if count == 0:
      return

    natural_height = self.pip_size * count
    overflow = natural_height > area.rect.height
    y_transform = (area.rect.height - self.pip_size) / (natural_height - self.pip_size) if overflow else 1
    offset_x = area.offset.x + area.rect.width / 2 - self.pip_size / 2

    offset_y = 0
    for i in range(0, count):
      offset_y = i * self.pip_size * y_transform

      self.canvas.create_oval(
        self.offset(offset_x, area.offset.y + offset_y * direction),
        self.offset(offset_x + self.pip_size, area.offset.y + (offset_y + self.pip_size) * direction),
        fill=choose_color(color == Color.White, "#ffffff", "#222222"),
        outline="#707070",
        tags=tags
      )

    # When overflow occurs, label the total pip count.
    if overflow:
      self.canvas.create_text(
        self.offset(
          offset_x + self.pip_size / 2,
          area.offset.y + (offset_y + self.pip_size / 2) * direction
        ),
        text=count,
        fill=choose_color(color != Color.White, "#ffffff", "#222222"),
        tags=tags,
      )

  def draw_point_pips(self, point_number, point):
    """Draw pips on a point
    """
    coord = self._point_coord(point_number)
    direction = self._point_direction(point_number)

    self.draw_pip_stack(
      point.count,
      point.color,
      direction,
      Area(
        Coord(coord.x, coord.y),
        Rect(self.point_width, self.point_height)
      ),
      [f"point:{point_number}"]
    )

  def draw_points(self):
    for i, point in enumerate(self.game.board.points):
      point_number = i + 1
      self.draw_point_board(point_number)
      self.draw_point_pips(point_number, point)


  def draw_bar(self):
    center_x = self.playable_width / 2
    bar_start_x = center_x - self.bar_width / 2

    # Draw the background of the bar
    self.canvas.create_rectangle(
      self.offset(bar_start_x, 0),
      self.offset(center_x + self.bar_width / 2, self.area.rect.height),
      fill="#656565",
      outline="",
      tags=["bar"],
    )

    # Draw the dice on the player's side
    dice_padding = 10
    def dice_offset(): # TODO would be nice to use _y_reflected
      if self.game.active_color == Color.White:
        return dice_padding
      else:
        return self.area.rect.height - dice_padding - self.die_size * 2

    for i in [0, 1]:
      self.canvas.create_rectangle(
        self.offset(
          center_x - self.die_size / 2,
          dice_offset() + self.die_size * i
        ),
        self.offset(
          center_x + self.die_size / 2,
          dice_offset() + self.die_size * (i + 1)
        ),
        fill=choose_color(self.game.active_color == Color.White)
      )
      self.canvas.create_text(
        self.offset(
          center_x,
          dice_offset() + self.die_size * (i + 0.5)
        ),
        text=self.game.active_dice.roll[i],
        fill=choose_color(self.game.active_color == Color.Black)
      )

    # Draw pips on the bar
    bar_pips_offset = dice_padding + self.die_size * 2 + self.doubling_cube_size + 10
    bar_pips_height = self.area.rect.height / 2 - bar_pips_offset - self.doubling_cube_size - 10
    for color in Color:
      direction = self._color_direction(color)
      self.draw_pip_stack(
        self.game.board.bar[color.value],
        color,
        direction,
        Area(
          Coord(bar_start_x, self._y_reflected(bar_pips_offset, direction)),
          Rect(self.bar_width, bar_pips_height)
        ),
        tags=["bar"]
      )

    # Draw the doubling cube in the center
    # TODO -- move appropriately for owning player
    self.canvas.create_rectangle(
      self.offset(
        center_x - self.doubling_cube_size / 2,
        self.area.rect.height / 2 - self.doubling_cube_size / 2
      ),
      self.offset(
        center_x + self.doubling_cube_size / 2,
        self.area.rect.height / 2 + self.doubling_cube_size / 2
      ),
      fill="#ffffff",
      tags=["doubling_cube"],
    )
    self.canvas.create_text(
      self.offset(
        center_x,
        self.area.rect.height / 2
      ),
      text=self.game.doubling_cube
    )

  def draw_off(self):
    # Draw the background of the off-board area
    off_start_x = self.playable_width
    self.canvas.create_rectangle(
      self.offset(off_start_x, 1),
      self.offset(off_start_x + self.off_width, self.area.rect.height),
      fill="#EAEAEA",
      outline="",
      tags=["off"],
    )

    separator_width = 3
    self.canvas.create_line(
      self.offset(off_start_x + separator_width / 2, 1),
      self.offset(off_start_x + separator_width / 2, self.area.rect.height),
      fill="#404040",
      width=separator_width
    )

    off_pips_x = off_start_x + separator_width
    off_pips_y = 10
    off_pips_height = self.area.rect.height / 3
    for color in Color:
      direction = self._color_direction(color) * -1
      self.draw_pip_stack(
        self.game.board.off[color.value],
        color,
        direction,
        Area(
          Coord(off_pips_x, self._y_reflected(off_pips_y, direction)),
          Rect(self.off_width - separator_width, off_pips_height)
        ),
        ["off"],
      )

  def draw(self):
    self.draw_chrome()
    self.draw_points()
    self.draw_bar()
    self.draw_off()
