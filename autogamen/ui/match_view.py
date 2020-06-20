from tkinter import Tk, Canvas, mainloop

from autogamen.game.types import Color
from .game_view import GameView
from .types import Area, Coord, Rect

class MatchView:
  WINDOW_WIDTH = 600
  WINDOW_HEIGHT = 400

  def __init__(self, match):
    self.match = match
    self.hooks = []

  def add_draw_hook(self, hook):
    self.hooks.append(hook)

  def create_window(self):
    self.tk = Tk()

    self.canvas = Canvas(
      self.tk,
      width=self.WINDOW_WIDTH,
      height=self.WINDOW_HEIGHT
    )
    self.canvas.pack(padx=0, pady=0)

  def draw_chrome(self):
    self.canvas.delete("all")

    # Draw the match info
    text = "; ".join([
      f"Match points: {self.match.point_goal}",
      f"White: {self.match.points[Color.White]}",
      f"Black: {self.match.points[Color.Black]}",
    ])
    self.canvas.create_text(
      self.WINDOW_WIDTH / 2,
      15,
      text=text
    )

    # Draw the game info
    text = "; ".join([
      f"Turn number: {self.match.current_game.turn_number}",
      f"White pips: {self.match.current_game.board.pip_count()[Color.White]}",
      f"Black pips: {self.match.current_game.board.pip_count()[Color.Black]}",
    ])
    self.canvas.create_text(
      self.WINDOW_WIDTH / 2,
      self.WINDOW_HEIGHT - 12,
      text=text
    )

    for hook in self.hooks:
      hook()

  def draw_game(self):
    padding = Coord(10, 30)
    self.game_view = GameView(
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
    self.game_view.draw()

  def run_tk_mainloop_once(self):
    self.tk.update_idletasks()
    self.tk.update()

  def draw(self):
    self.draw_chrome()
    self.draw_game()

  def run(self):
    self.match.start_game()
    while self.match.winner is None:
      self.draw()
      self.run_tk_mainloop_once()
      if self.match.tick():
        self.match.start_game()


def display_board(board):
  """crude function to display a board for debugging purposes"""
  from autogamen.ai.bozo import BozoPlayer
  from autogamen.game.match import Match
  from autogamen.game.types import Color

  match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)])
  match.start()
  match.current_game.board = board
  match_view = MatchView(match)
  match_view.create_window()
  match_view.draw_chrome()
  match_view.draw_board()
  mainloop()
