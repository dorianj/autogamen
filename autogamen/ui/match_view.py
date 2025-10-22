from tkinter import Canvas, Tk, mainloop
from typing import Any

from autogamen.game.game_types import Color

from .game_view import Area, Coord, GameView, Rect


class MatchView:
  WINDOW_WIDTH = 600
  WINDOW_HEIGHT = 400

  def __init__(self, match: Any) -> None:
    self.match = match

  def create_window(self) -> None:
    self.tk = Tk()

    self.canvas = Canvas(
      self.tk,
      width=self.WINDOW_WIDTH,
      height=self.WINDOW_HEIGHT
    )
    self.canvas.pack(padx=0, pady=0)

  def draw_chrome(self) -> None:
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

  def draw_game(self) -> None:
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

  def run_tk_mainloop_once(self) -> None:
    self.tk.update_idletasks()
    self.tk.update()

  def draw(self) -> None:
    self.draw_chrome()
    self.draw_game()

  def run(self) -> None:
    while self.match.winner is None:
      self.match.pre_tick()
      self.draw()
      self.run_tk_mainloop_once()
      self.match.tick()


def display_board(board: Any) -> None:
  """crude function to display a board for debugging purposes"""
  from autogamen.ai.players import BozoPlayer  # noqa: PLC0415
  from autogamen.game.game_types import Color  # noqa: PLC0415
  from autogamen.game.match import Match  # noqa: PLC0415

  match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)], point_goal=1)
  match.pre_tick()
  assert match.current_game is not None
  match.current_game.board = board
  match_view = MatchView(match)
  match_view.create_window()
  match_view.draw_chrome()
  match_view.draw_game()
  mainloop()
