import time

from autogamen.game.player import Player
from autogamen.game.types import Move, TurnAction

from .types import Coord


class HumanPlayer(Player):
  # All moves this player could do on this turn
  all_possible_moves = []

  # After user clicks on a point, set it to this.
  half_selected_point = None

  # Moves that have been selected
  selected_moves = tuple()
  have_complete_turn_selected = False

  # Board as it was before this turn started
  board_before_turn = None

  def extract_point(self, point_tag):
    if not point_tag:
      return None

    parts = point_tag.split(":")
    if len(parts) == 2 and parts[0] == "point":
      return int(parts[1])

    return None

  def tags_at_coord(self, coord):
    closest = self.match_view.canvas.find_closest(coord.x, coord.y)

    if not closest:
      return None

    return self.match_view.canvas.gettags(closest)

  def target_for_tags(self, tags):
    for tag in tags:
      if tag == "bar":
        return "bar"
      elif tag == "off":
        return "off"
      else:
        point_number = self.extract_point(tag)
        if point_number:
          return point_number

    return None

  def have_complete_turn_selected(self):
    return tuple(self.selected_moves) in self.all_possible_moves

  def remaining_moves(self):
    # Return generator individual moves that are possible given self.selected_moves
    for move in self.all_possible_moves:
      if len(move) > len(self.selected_moves) and move[:len(self.selected_moves)] == self.selected_moves:
        yield move[len(self.selected_moves)]


  def draw_turn_selector(self, coord):
    tags = ["turn_selector"]
    # Add a rectangle to the click point
    self.match_view.canvas.create_rectangle(
      coord.x - 10, coord.y - 10,
      coord.x + 10, coord.y + 10,
      fill="blue",
      tags=tags + ["source"]
    )

    # Add a rectangle at all destinations
    dest_color = "green"
    gv = self.match_view.game_view
    for allowed_move in self.allowed_moves_from_target(self.half_selected_point):
      if allowed_move.destination_is_off:
        self.match_view.canvas.create_rectangle(
          gv.offset(gv.playable_width, 1),
          gv.offset(gv.playable_width + gv.off_width, gv.area.rect.height),
          fill=dest_color,
          tags=tags + ["off"],
        )
      else:
        point_coord = gv._point_coord(allowed_move.destination_point_number)
        direction = gv._point_direction(allowed_move.destination_point_number)
        indicator_height = gv.point_height / 4

        self.match_view.canvas.create_rectangle(
          gv.offset(point_coord.x, point_coord.y + indicator_height * direction),
          gv.offset(point_coord.x + gv.point_width, point_coord.y + (indicator_height*2) * direction),
          fill=dest_color,
          tags=tags + [f"point:{allowed_move.destination_point_number}"],
        )

  def remove_from_canvas_by_tag(self, tag):
    for item in self.match_view.canvas.find_withtag(tag):
      self.match_view.canvas.delete(item)

  def clear_half_turn_selector(self):
    self.remove_from_canvas_by_tag("turn_selector")

  def allowed_moves_from_target(self, target):
    # Find moves with this point as a starting point
    def _filter_move(move):
      return (
        (move.point_number == Move.Bar and target == "bar") or
        (move.point_number == target)
      )

    return list(filter(_filter_move, set(self.remaining_moves())))

  def complete_half_move(self, move):
    self.half_selected_point = None
    self.selected_moves += (move,)
    self.clear_half_turn_selector()
    self.match_view.match.current_game.board = self.board_before_turn.copy_apply_moves(self.selected_moves)
    self.match_view.draw()

    if self.have_complete_turn_selected():
      self.match_view.canvas.create_rectangle(
        10, 5,
        50, 25,
        fill="green",
        tags=["turn_completer", "finish"],
      )

    if not self.match_view.canvas.find_withtag("undo"):
      self.match_view.canvas.create_rectangle(
        60, 5,
        90, 25,
        fill="red",
        tags=["turn_completer", "undo"],
      )

  def complete_turn(self):
    self.turn_finished = True
    self.remove_from_canvas_by_tag("turn_completer")

  def mouse_click(self, click):
    click_coord = Coord(click.x, click.y)
    target_tags = self.tags_at_coord(click_coord)

    click_target = self.target_for_tags(target_tags)
    target_is_off = click_target == "off"
    target_is_point = type(click_target) == int
    target_is_bar = click_target == "bar"

    print(f"target_tags: {target_tags}; click_target: {click_target}")

    if "undo" in target_tags:
      self.selected_moves = tuple()
      self.match_view.match.current_game.board = self.board_before_turn.mutable_copy()
      self.match_view.draw()
    elif self.have_complete_turn_selected():
      if "finish" in target_tags:
        self.complete_turn()
    elif self.half_selected_point:
      if "source" in target_tags:
        self.half_selected_point = None
        self.clear_half_turn_selector()
      elif target_is_off or target_is_point:
        # Possibly valid move
        for move in self.allowed_moves_from_target(self.half_selected_point):
          destination_matches = (
            (target_is_off and move.destination_is_off) or
            (target_is_point and move.destination_point_number == click_target)
          )
          if destination_matches:
            self.complete_half_move(move)
    else:
      if (target_is_bar or target_is_point) and len(self.allowed_moves_from_target(click_target)) > 0:
        self.half_selected_point = click_target
        self.draw_turn_selector(click_coord)
      else:
        self.half_selected_point = None

  def attach(self, match_view):
    self.match_view = match_view
    self.match_view.canvas.bind('<Button>', self.mouse_click)

  def action(self, possible_moves):
    if not len(possible_moves):
      return [TurnAction.Pass]

    # Run the mainloop until we have an event.
    self.all_possible_moves = [move for move, board in possible_moves]
    self.board_before_turn = self.match_view.match.current_game.board.frozen_copy()
    self.selected_moves = tuple()
    self.turn_finished = False
    while not self.turn_finished:
      self.match_view.run_tk_mainloop_once()
      time.sleep(1/15.0)

    self.match_view.match.current_game.board = self.board_before_turn.mutable_copy()
    return [TurnAction.Move, self.selected_moves]

  def accept_doubling_cube(self):
    return False
