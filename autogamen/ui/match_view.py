from tkinter import Tk, Canvas, mainloop

from .board_view import BoardView
from .types import Area, Coord, Rectangle

class MatchView:
  WINDOW_WIDTH = 600
  WINDOW_HEIGHT = 400

  def __init__(self, match):
    self.match = match

  def create_window(self):
    master = Tk()

    self.canvas = Canvas(
      master,
      width=self.WINDOW_WIDTH,
      height=self.WINDOW_HEIGHT
    )
    self.canvas.pack(padx=0, pady=0)

  def draw_chrome(self):
    pass

  def draw_board(self):
    padding = Coord(50, 10)
    bv = BoardView(
      self.canvas,
      Area(
        padding,
        Rectangle(
          self.WINDOW_WIDTH - padding.x * 2,
          self.WINDOW_HEIGHT - padding.y * 2,
        )
      ),
      self.match.current_game
    )

    bv.draw()


  def run(self):
    self.draw_chrome()
    self.draw_board()
    mainloop()

