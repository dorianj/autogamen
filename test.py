from autogamen.ui.match_view import MatchView
from autogamen.game.match import Match


match = Match()
match_view = MatchView(match)
match_view.create_window()
match_view.run()
