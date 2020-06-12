from math import floor

from .types import Coord
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
    self.point_width = floor((self.area.rectangle.width - self.bar_width) / 12.0)
    self.point_height = floor(self.area.rectangle.height / 2 * 0.85)
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
      self.offset(self.area.rectangle.width, self.area.rectangle.height),
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
      y_pos = self.area.rectangle.height

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

  def draw_point_pips(self, point_number, point):
    coord = self._point_coord(point_number)
    top_sign = self._point_sign(point_number)
    start_x = coord.x + self.point_width / 2 - self.pip_size / 2
    overflow = point.count > self.max_displayed_pips
    height_transform = self.point_height / (self.pip_size * point.count) if overflow else 1

    for i in range(0, point.count):
      offset_y = 0
      offset_y += i * self.pip_size * height_transform

      self.canvas.create_oval(
        self.offset(start_x, coord.y + offset_y * top_sign),
        self.offset(start_x + self.pip_size, coord.y + (offset_y + self.pip_size) * top_sign),
        fill=choose_color(point.color == Color.White, "#ffffff", "#222222"),
        outline="#707070"
      )

      # For the last pip, if it overflowed, label the total pip count
      if overflow and i == point.count - 1:
        self.canvas.create_text(
          self.offset(
            start_x + self.pip_size / 2,
            coord.y + (offset_y + self.pip_size / 2) * top_sign
          ),
          text=point.count,
          fill=choose_color(point.color == Color.White, "#ffffff", "#222222")
          )

  def draw_points(self):
    for i, point in enumerate(self.game.board.points):
      point_number = i + 1
      self.draw_point_board(point_number)
      self.draw_point_pips(point_number, point)


  def draw_bar(self):
    center_x = self.area.rectangle.width / 2

    # Draw the background of the bar
    self.canvas.create_rectangle(
      self.offset(center_x - self.bar_width / 2, 0),
      self.offset(center_x + self.bar_width / 2, self.area.rectangle.height),
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
        return self.area.rectangle.height - dice_padding - self.die_size * 2

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

    # Draw the doubling cube in the center
    # TODO -- move appropriately
    self.canvas.create_rectangle(
      self.offset(
        center_x - self.doubling_cube_size / 2,
        self.area.rectangle.height / 2
      ),
      self.offset(
        center_x + self.doubling_cube_size / 2,
        self.area.rectangle.height / 2 + self.doubling_cube_size
      ),
      fill="#ffffff"
    )
    self.canvas.create_text(
      self.offset(
        center_x,
        self.area.rectangle.height / 2 + self.doubling_cube_size / 2
      ),
      text=self.game.doubling_cube
    )



  def draw(self):
    self.draw_chrome()
    self.draw_points()
    self.draw_bar()
