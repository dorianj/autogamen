from math import floor

from .types import Area, Coord, Rect
from autogamen.game.types import Color


def choose_color(white, white_color="#ffffff", black_color="#000000"):
  """Ternary statement with defaults."""
  return white_color if white else black_color

class BoardView:
  def __init__(self, canvas, area, game):
    self.canvas = canvas
    self.area = area
    self.game = game


    self.bar_width = 40
    self.point_width = floor((self.area.rect.width - self.bar_width) / 12.0)
    self.point_height = floor(self.area.rect.height / 2 * 0.85)
    self.max_displayed_pips = 5
    self.pip_size = floor(min(
      self.point_height / self.max_displayed_pips,
      self.point_width * 0.95
    ))
    self.doubling_cube_size = 25
    self.die_size = 18

  def offset(self, *args):
    """Returns the offset point. Accepts x/y args or a single Coord"""
    point = args[0] if len(args) == 1 else Coord(args[0], args[1])
    return Coord(point.x + self.area.offset.x, point.y + self.area.offset.y)

  def draw_chrome(self):
    # Draw a border around the board
    self.canvas.create_rectangle(
      self.offset(0, 0),
      self.offset(self.area.rect.width, self.area.rect.height),
    )

  def _point_coord(self, point_number):
    """Returns the Coord coordinate for a given point number"""
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

  def _point_sign(self, point_number):
    return -1 if point_number <= 12 else 1

  def draw_point_board(self, point_number):
    coord = self._point_coord(point_number)
    top_sign = self._point_sign(point_number)

    # Draw the triangle
    self.canvas.create_polygon(
      self.offset(coord.x, coord.y),
      self.offset(coord.x + self.point_width, coord.y),
      self.offset(coord.x + self.point_width / 2, coord.y + self.point_height * top_sign),
      fill=choose_color(point_number % 2, "#990000", "#A5A5A5")
    )

    # Draw the point number indicator
    self.canvas.create_text(
      self.area.offset.x + coord.x + self.point_width / 2, self.area.offset.y + coord.y + 10 * top_sign,
      text=point_number
    )

  def draw_pip_stack(self, count, color, direction, area, max_count):
    """Draw a stack of pips
      :count: int, number of pips to draw
      :color: Color
      :direction: int, 1 for up-to-down, -1 for down-to-up
      :area: Area, bounding box. Coord is top left for downwards, bottom left for upwards
      :max_count: max number of pips to render before stacking (TODO: infer from area)
    """
    overflow = count > max_count
    height_transform = area.rect.height / (self.pip_size * count) if overflow else 1

    for i in range(0, count):
      offset_y = 0
      offset_y += i * self.pip_size * height_transform

      self.canvas.create_oval(
        self.offset(area.offset.x, area.offset.y + offset_y * direction),
        self.offset(area.offset.x + self.pip_size, area.offset.y + (offset_y + self.pip_size) * direction),
        fill=choose_color(color == Color.White, "#ffffff", "#222222"),
        outline="#707070"
      )

      # For the last pip, if it overflowed, label the total pip count
      if overflow and i == count - 1:
        self.canvas.create_text(
          self.offset(
            area.offset.x + self.pip_size / 2,
            area.offset.y + (offset_y + self.pip_size / 2) * direction
          ),
          text=count,
          fill=choose_color(color == Color.White, "#ffffff", "#222222")
          )


  def draw_point_pips(self, point_number, point):
    """Draw pips on a point
    """
    coord = self._point_coord(point_number)
    top_sign = self._point_sign(point_number)
    start_x = coord.x + self.point_width / 2 - self.pip_size / 2

    self.draw_pip_stack(
      point.count,
      point.color,
      top_sign,
      Area(
        Coord(start_x, coord.y),
        Rect(self.point_width, self.point_height)
      ),
      self.max_displayed_pips
    )

  def draw_points(self):
    for i, point in enumerate(self.game.board.points):
      point_number = i + 1
      self.draw_point_board(point_number)
      self.draw_point_pips(point_number, point)


  def draw_bar(self):
    center_x = self.area.rect.width / 2

    # Draw the background of the bar
    self.canvas.create_rectangle(
      self.offset(center_x - self.bar_width / 2, 0),
      self.offset(center_x + self.bar_width / 2, self.area.rect.height),
      fill="#656565",
      outline="",
    )

    # Draw the dice on the player's side
    dice_padding = 10
    dice_offset = 0
    def dice_offset():
      if self.game.white_is_active():
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
        fill=choose_color(self.game.white_is_active())
      )
      self.canvas.create_text(
        self.offset(
          center_x,
          dice_offset() + self.die_size * (i + 0.5)
        ),
        text=self.game.active_dice.roll[i],
        fill=choose_color(not self.game.white_is_active())
      )

    # Draw pips on the bar


    # Draw the doubling cube in the center
    # TODO -- move appropriately for owning player
    self.canvas.create_rectangle(
      self.offset(
        center_x - self.doubling_cube_size / 2,
        self.area.rect.height / 2
      ),
      self.offset(
        center_x + self.doubling_cube_size / 2,
        self.area.rect.height / 2 + self.doubling_cube_size
      ),
      fill="#ffffff"
    )
    self.canvas.create_text(
      self.offset(
        center_x,
        self.area.rect.height / 2 + self.doubling_cube_size / 2
      ),
      text=self.game.doubling_cube
    )



  def draw(self):
    self.draw_chrome()
    self.draw_points()
    self.draw_bar()
