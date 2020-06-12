from autogamen.ui.match_view import MatchView
from autogamen.game.match import Match

from autogamen.ai.bozo import BozoPlayer
from autogamen.game.types import Color

match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)])
match_view = MatchView(match)
match_view.create_window()
match_view.run()
