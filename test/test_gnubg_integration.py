"""integration tests for gnubg interface and player.

these tests verify that our gnubg integration works correctly, including:
- board state conversion to gnubg format
- move notation parsing from gnubg to our format
- actual gameplay with gnubg making moves
"""
import unittest

from autogamen.ai.players import BozoPlayer, GnubgPlayer
from autogamen.game.board import Board
from autogamen.game.game import Game
from autogamen.game.game_types import Color, Dice, Move, Point
from autogamen.gnubg.interface import GnubgInterface


def standard_starting_points():
    return [
        Point(2, Color.White),
        Point(),
        Point(),
        Point(),
        Point(),
        Point(5, Color.Black),
        Point(),
        Point(3, Color.Black),
        Point(),
        Point(),
        Point(),
        Point(5, Color.White),
        Point(5, Color.Black),
        Point(),
        Point(),
        Point(),
        Point(3, Color.White),
        Point(),
        Point(5, Color.White),
        Point(),
        Point(),
        Point(),
        Point(),
        Point(2, Color.Black),
    ]


class TestGnubgBoardConversion(unittest.TestCase):
    """test that board conversion to gnubg format is correct.

    this would have caught the bug where point numbering was backwards for white.
    """

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.gnubg.start()
        self.board = Board(standard_starting_points())

    def tearDown(self):
        self.gnubg.stop()

    def test_white_board_conversion(self):
        """for white, gnubg point numbering is flipped from ours."""
        gnubg_str = self.gnubg._board_to_gnubg_simple(self.board, Color.White)
        parts = gnubg_str.split()

        # gnubg point 1 = our point 24 = 2 black checkers = -2
        self.assertEqual(parts[1], "-2", "gnubg point 1 should be our point 24 (2 black)")

        # gnubg point 6 = our point 19 = 5 white checkers = +5
        self.assertEqual(parts[6], "5", "gnubg point 6 should be our point 19 (5 white)")

        # gnubg point 8 = our point 17 = 3 white checkers = +3
        self.assertEqual(parts[8], "3", "gnubg point 8 should be our point 17 (3 white)")

        # gnubg point 12 = our point 13 = 5 black checkers = -5
        self.assertEqual(parts[12], "-5", "gnubg point 12 should be our point 13 (5 black)")

        # gnubg point 13 = our point 12 = 5 white checkers = +5
        self.assertEqual(parts[13], "5", "gnubg point 13 should be our point 12 (5 white)")

        # gnubg point 17 = our point 8 = 3 black checkers = -3
        self.assertEqual(parts[17], "-3", "gnubg point 17 should be our point 8 (3 black)")

        # gnubg point 19 = our point 6 = 5 black checkers = -5
        self.assertEqual(parts[19], "-5", "gnubg point 19 should be our point 6 (5 black)")

        # gnubg point 24 = our point 1 = 2 white checkers = +2
        self.assertEqual(parts[24], "2", "gnubg point 24 should be our point 1 (2 white)")

    def test_black_board_conversion(self):
        """for black, gnubg point numbering is same as ours."""
        gnubg_str = self.gnubg._board_to_gnubg_simple(self.board, Color.Black)
        parts = gnubg_str.split()

        # gnubg point 1 = our point 1 = 2 white checkers = -2
        self.assertEqual(parts[1], "-2", "gnubg point 1 should be our point 1 (2 white)")

        # gnubg point 6 = our point 6 = 5 black checkers = +5
        self.assertEqual(parts[6], "5", "gnubg point 6 should be our point 6 (5 black)")

        # gnubg point 8 = our point 8 = 3 black checkers = +3
        self.assertEqual(parts[8], "3", "gnubg point 8 should be our point 8 (3 black)")

        # gnubg point 24 = our point 24 = 2 black checkers = +2
        self.assertEqual(parts[24], "2", "gnubg point 24 should be our point 24 (2 black)")


class TestGnubgHints(unittest.TestCase):
    """test that gnubg returns reasonable hints for standard positions."""

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.gnubg.start()
        self.board = Board(standard_starting_points())

    def tearDown(self):
        self.gnubg.stop()

    def test_white_gets_hints_from_starting_position(self):
        """gnubg should return at least one hint for white from starting position."""
        hints = self.gnubg.get_hint(self.board, Color.White, (3, 1))
        self.assertGreater(len(hints), 0, "gnubg should return hints for white")

    def test_black_gets_hints_from_starting_position(self):
        """gnubg should return at least one hint for black from starting position."""
        hints = self.gnubg.get_hint(self.board, Color.Black, (3, 1))
        self.assertGreater(len(hints), 0, "gnubg should return hints for black")

    def test_hints_have_valid_equity(self):
        """gnubg hints should have equity values in reasonable range."""
        hints = self.gnubg.get_hint(self.board, Color.White, (3, 1))
        self.assertGreater(len(hints), 0)
        # equity for opening moves should be roughly in [-1, 1] range
        self.assertGreater(hints[0].equity, -2.0)
        self.assertLess(hints[0].equity, 2.0)


class TestGnubgMoveNotationParsing(unittest.TestCase):
    """test parsing of gnubg move notation to our move format."""

    def setUp(self):
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))

        self.black_player = GnubgPlayer(Color.Black, plies=0)
        self.black_player.start_game(Game([BozoPlayer(Color.White), self.black_player]))

    def tearDown(self):
        self.white_player.gnubg.stop()
        self.black_player.gnubg.stop()

    def test_white_simple_move_parsing(self):
        """parse a simple move for white: 24/21"""
        # gnubg "24/21" for white = our point 1 moving distance 3
        moves = self.white_player._parse_gnubg_move_string("24/21")
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].color, Color.White)
        self.assertEqual(moves[0].point_number, 1)
        self.assertEqual(moves[0].distance, 3)

    def test_white_compound_move_parsing(self):
        """parse compound move for white: 13/10 6/5"""
        # gnubg "13/10" = our point 12 moving distance 3
        # gnubg "6/5" = our point 19 moving distance 1
        moves = self.white_player._parse_gnubg_move_string("13/10 6/5")
        self.assertEqual(len(moves), 2)

        # moves can be in either order
        move_set = set(moves)
        self.assertIn(Move(Color.White, 12, 3), move_set)
        self.assertIn(Move(Color.White, 19, 1), move_set)

    def test_black_simple_move_parsing(self):
        """parse a simple move for black: 1/4"""
        # for black, gnubg numbering matches ours
        moves = self.black_player._parse_gnubg_move_string("1/4")
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].color, Color.Black)
        self.assertEqual(moves[0].point_number, 1)
        self.assertEqual(moves[0].distance, 3)

    def test_black_compound_move_parsing(self):
        """parse compound move for black: 13/10 6/5"""
        moves = self.black_player._parse_gnubg_move_string("13/10 6/5")
        self.assertEqual(len(moves), 2)

        move_set = set(moves)
        self.assertIn(Move(Color.Black, 13, 3), move_set)
        self.assertIn(Move(Color.Black, 6, 1), move_set)


class TestGnubgPlayerGameplay(unittest.TestCase):
    """test that gnubg player can actually play games.

    this tests the full integration: board conversion, hint retrieval,
    move parsing, and move selection.
    """

    def test_gnubg_can_complete_game_vs_random(self):
        """gnubg should be able to complete a full game against random player."""
        gnubg = GnubgPlayer(Color.White, plies=0)
        random = BozoPlayer(Color.Black)

        game = Game([gnubg, random])
        game.start()

        max_turns = 500  # prevent infinite loops
        turn_count = 0

        while game.winner is None and turn_count < max_turns:
            game.run_turn()
            turn_count += 1

        # game should finish (someone wins)
        self.assertIsNotNone(game.winner, f"game should complete within {max_turns} turns")

        gnubg.gnubg.stop()

    def test_two_gnubg_players_can_complete_game(self):
        """two gnubg players should be able to play against each other."""
        gnubg_white = GnubgPlayer(Color.White, plies=0)
        gnubg_black = GnubgPlayer(Color.Black, plies=0)

        game = Game([gnubg_white, gnubg_black])
        game.start()

        max_turns = 500
        turn_count = 0

        while game.winner is None and turn_count < max_turns:
            game.run_turn()
            turn_count += 1

        self.assertIsNotNone(game.winner, f"game should complete within {max_turns} turns")

        gnubg_white.gnubg.stop()
        gnubg_black.gnubg.stop()


class TestGnubgStrengthLevels(unittest.TestCase):
    """test that different gnubg strength levels work."""

    def test_gnubg_accepts_different_ply_settings(self):
        """gnubg should accept different ply settings from 0 to 4.

        note: 7-ply is world class but very slow, excluded from quick tests
        """
        for plies in [0, 1, 2, 4]:
            gnubg = GnubgInterface(plies=plies)
            gnubg.start()

            board = Board(standard_starting_points())
            hints = gnubg.get_hint(board, Color.White, (3, 1))

            self.assertGreater(len(hints), 0, f"gnubg with {plies} plies should return hints")

            gnubg.stop()


class TestGnubgMoveMatching(unittest.TestCase):
    """test that gnubg suggestions match our possible_moves output.

    this is the critical integration test: if gnubg suggests moves that aren't
    in our possible_moves set, the player falls back to random moves.
    """

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.gnubg.start()
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))
        self.black_player = GnubgPlayer(Color.Black, plies=0)
        self.black_player.start_game(Game([BozoPlayer(Color.White), self.black_player]))

    def tearDown(self):
        self.gnubg.stop()
        self.white_player.gnubg.stop()
        self.black_player.gnubg.stop()

    def test_white_opening_move_is_in_possible_moves(self):
        """gnubg's top suggestion for white opening should be in our possible_moves."""
        board = Board(standard_starting_points())
        dice_roll = (3, 1)

        # get gnubg's suggestion
        hints = self.gnubg.get_hint(board, Color.White, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints")

        # parse it to our format
        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves)

        # get our possible moves
        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.White, dice)

        # verify gnubg's suggestion is in our set
        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' (parsed as {our_moves}) "
            f"but it's not in our {len(possible)} possible moves")

    def test_black_opening_move_is_in_possible_moves(self):
        """gnubg's top suggestion for black opening should be in our possible_moves."""
        board = Board(standard_starting_points())
        dice_roll = (4, 2)

        hints = self.gnubg.get_hint(board, Color.Black, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints")

        our_moves = self.black_player._parse_gnubg_move_string(hints[0].moves)

        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.Black, dice)

        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' (parsed as {our_moves}) "
            f"but it's not in our {len(possible)} possible moves")

    def test_doubles_move_is_in_possible_moves(self):
        """gnubg suggestions for doubles should be in our possible_moves."""
        board = Board(standard_starting_points())
        dice_roll = (2, 2)

        hints = self.gnubg.get_hint(board, Color.White, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints for doubles")

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves)

        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.White, dice)

        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' (parsed as {our_moves}) "
            f"but it's not in our {len(possible)} possible moves for doubles")


class TestGnubgMidGamePositions(unittest.TestCase):
    """test gnubg integration with various mid-game positions."""

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.gnubg.start()
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))

    def tearDown(self):
        self.gnubg.stop()
        self.white_player.gnubg.stop()

    def test_bearing_off_position(self):
        """test gnubg suggestions for bearing off position."""
        # white is bearing off: all checkers in home board
        points = [Point() for _ in range(18)]
        points.extend([
            Point(4, Color.White),  # point 19
            Point(3, Color.White),  # point 20
            Point(4, Color.White),  # point 21
            Point(2, Color.White),  # point 22
            Point(2, Color.White),  # point 23
            Point(),                 # point 24
        ])

        board = Board(points)
        dice_roll = (6, 3)

        hints = self.gnubg.get_hint(board, Color.White, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints for bearing off")

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves)

        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.White, dice)

        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' for bearing off "
            f"but it's not in our possible moves")

    def test_bar_position(self):
        """test gnubg suggestions when player has checker on bar."""
        # white has a checker on the bar
        points = list(standard_starting_points())
        points[0] = Point(1, Color.White)  # reduce point 1 by 1

        board = Board(points, bar=[1, 0])  # white has 1 on bar
        dice_roll = (5, 3)

        hints = self.gnubg.get_hint(board, Color.White, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints with checker on bar")

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves)

        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.White, dice)

        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' for bar position "
            f"but it's not in our possible moves")

    def test_racing_position(self):
        """test gnubg suggestions in a racing position (no contact)."""
        # simplified racing position
        points = [
            Point(3, Color.White),   # point 1
            Point(2, Color.White),   # point 2
            Point(3, Color.White),   # point 3
            Point(2, Color.White),   # point 4
            Point(3, Color.White),   # point 5
            Point(2, Color.White),   # point 6
        ]
        points.extend([Point() for _ in range(12)])
        points.extend([
            Point(2, Color.Black),   # point 19
            Point(3, Color.Black),   # point 20
            Point(2, Color.Black),   # point 21
            Point(3, Color.Black),   # point 22
            Point(2, Color.Black),   # point 23
            Point(3, Color.Black),   # point 24
        ])

        board = Board(points)
        dice_roll = (6, 4)

        hints = self.gnubg.get_hint(board, Color.White, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints for racing")

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves)

        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.White, dice)

        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' for racing position "
            f"but it's not in our possible moves")


if __name__ == "__main__":
    unittest.main()
