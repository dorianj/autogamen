from tkinter import Tk, Canvas, mainloop

from .board_view import BoardView
from .types import Area, Coord, Rect

class MatchView:
  WINDOW_WIDTH = 600
  WINDOW_HEIGHT = 400

  def __init__(self, match):
    self.match = match

  def create_window(self):
    self.tk = Tk()

    self.canvas = Canvas(
      self.tk,
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
        Rect(
          self.WINDOW_WIDTH - padding.x * 2,
          self.WINDOW_HEIGHT - padding.y * 2,
        )
      ),
      self.match.current_game
    )
    self.canvas.delete("all")
    bv.draw()


  def run(self):
    self.match.start()
    while True:
      self.draw_chrome()
      self.draw_board()
      self.tk.update_idletasks()
      self.tk.update()
      self.match.tick()



def display_board(board):
  """crude function to display a board for debugging purposes"""
  from autogamen.ai.bozo import BozoPlayer
  from autogamen.game.match import Match
  from autogamen.game.types import Color

  match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)])
  match.current_game.board = board
  match_view = MatchView(match)
  match_view.create_window()
  match_view.run()
