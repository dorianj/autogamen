"""UI match runner - migrated from run_ui_match.py"""

from autogamen.ai.simple import DeltaPlayer
from autogamen.game.game_types import Color
from autogamen.game.match import Match
from autogamen.ui.match_view import MatchView
from autogamen.ui.ui_player import HumanPlayer


def run_ui_match(opponent: str, points: int) -> None:
    """Main entry point for UI matches"""
    # TODO: make opponent selection more flexible
    # For now, hardcode DeltaPlayer
    opponent_cls = DeltaPlayer

    human_player = HumanPlayer(Color.White)
    match = Match([human_player, opponent_cls(Color.Black)], points)
    match_view = MatchView(match)
    match_view.create_window()
    human_player.attach(match_view)
    match_view.run()
