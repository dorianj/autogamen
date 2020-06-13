from autogamen.ai.bozo import BozoPlayer
from autogamen.game.match import Match
from autogamen.game.types import Color
from autogamen.ui.match_view import MatchView

match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)])
match_view = MatchView(match)
match_view.create_window()
match_view.run()
