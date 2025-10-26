"""integration tests for gnubg interface and player.

these tests verify that our gnubg integration works correctly, including:
- board state conversion to gnubg format
- move notation parsing from gnubg to our format
- actual gameplay with gnubg making moves
"""
import time
import unittest

from autogamen.ai.players import BozoPlayer, GnubgPlayer
from autogamen.game.board import Board
from autogamen.game.game import Game
from autogamen.game.game_types import Color, Dice, Move, Point
from autogamen.gnubg.interface import GnubgInterface, get_daemon


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


class TestGnubgSocketInterface(unittest.TestCase):
    """test the socket-based gnubg interface."""

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.board = Board(standard_starting_points())

    def test_socket_connection_works(self):
        """verify socket connection to gnubg daemon works."""
        # get hint should establish connection and return results
        hints = self.gnubg.get_hint(self.board, Color.White, (3, 1))
        self.assertGreater(len(hints), 0, "should get hints from socket interface")
        self.assertIsNotNone(hints[0].moves, "hint should have move string")

    def test_multiple_requests_same_connection(self):
        """test that multiple requests work on the same interface instance."""
        # first request
        hints1 = self.gnubg.get_hint(self.board, Color.White, (3, 1))
        self.assertGreater(len(hints1), 0)

        # second request with different dice
        hints2 = self.gnubg.get_hint(self.board, Color.White, (6, 4))
        self.assertGreater(len(hints2), 0)

        # third request for different color
        hints3 = self.gnubg.get_hint(self.board, Color.Black, (2, 2))
        self.assertGreater(len(hints3), 0)

    def test_handles_doubles_correctly(self):
        """test that doubles dice are handled correctly in socket interface."""
        hints = self.gnubg.get_hint(self.board, Color.White, (2, 2))
        # sometimes connection fails, retry once
        if not hints:
            time.sleep(1)
            hints = self.gnubg.get_hint(self.board, Color.White, (2, 2))

        self.assertGreater(len(hints), 0, "should get hints for doubles")
        # gnubg often uses notation like "24/22(2)" for doubles
        # just verify we get some response
        self.assertIsNotNone(hints[0].moves)


class TestGnubgFibsFormat(unittest.TestCase):
    """test FIBS board format conversion for socket interface."""

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.board = Board(standard_starting_points())

    def test_fibs_format_has_correct_field_count(self):
        """FIBS format should have the right number of colon-separated fields."""
        fibs_str = self.gnubg._board_to_fibs(self.board, Color.White, (3, 1))
        fields = fibs_str.split(':')

        # minimum FIBS fields: board, player, opponent, match info, points, dice, etc.
        self.assertGreater(len(fields), 40, f"FIBS format should have many fields, got {len(fields)}")

    def test_fibs_format_for_white(self):
        """test FIBS format generation for white player."""
        fibs_str = self.gnubg._board_to_fibs(self.board, Color.White, (6, 3))
        fields = fibs_str.split(':')

        # basic field checks
        self.assertEqual(fields[0], "board", "first field should be 'board'")
        self.assertEqual(fields[1], "You", "player name")
        self.assertEqual(fields[2], "opponent", "opponent name")

        # dice fields are at indices 33-36
        self.assertEqual(fields[33], "6", "first die")
        self.assertEqual(fields[34], "3", "second die")
        self.assertEqual(fields[35], "0", "third die (0 for non-doubles)")
        self.assertEqual(fields[36], "0", "fourth die (0 for non-doubles)")

    def test_fibs_format_for_black(self):
        """test FIBS format generation for black player."""
        fibs_str = self.gnubg._board_to_fibs(self.board, Color.Black, (4, 2))
        fields = fibs_str.split(':')

        # basic checks
        self.assertEqual(fields[0], "board")

        # dice at indices 33-36
        self.assertEqual(fields[33], "4", "first die")
        self.assertEqual(fields[34], "2", "second die")

        # color field should be -1 for black (index 41)
        self.assertEqual(fields[41], "-1", "color should be -1 for black")
        # just verify we get a valid FIBS string
        self.assertGreater(len(fields), 40)

    def test_fibs_format_with_doubles(self):
        """test FIBS format with doubles dice."""
        fibs_str = self.gnubg._board_to_fibs(self.board, Color.White, (3, 3))
        fields = fibs_str.split(':')

        # for doubles, all four dice fields should have the same value (indices 33-36)
        self.assertEqual(fields[33], "3", "first die")
        self.assertEqual(fields[34], "3", "second die")
        self.assertEqual(fields[35], "3", "third die")
        self.assertEqual(fields[36], "3", "fourth die")

    def test_fibs_format_with_bar(self):
        """test FIBS format with checkers on the bar."""
        # create board with pieces on bar
        points = list(standard_starting_points())
        points[0] = Point(1, Color.White)  # reduce point 1 by 1
        board_with_bar = Board(points, bar=[1, 0])  # white has 1 on bar

        fibs_str = self.gnubg._board_to_fibs(board_with_bar, Color.White, (5, 3))
        fields = fibs_str.split(':')

        # verify we get valid FIBS format (exact bar position varies by format version)
        self.assertGreater(len(fields), 40, "should have valid FIBS format with bar")


class TestGnubgDaemon(unittest.TestCase):
    """test gnubg daemon management."""

    def test_daemon_singleton(self):
        """test that daemon is a singleton."""
        daemon1 = get_daemon()
        daemon2 = get_daemon()

        self.assertIs(daemon1, daemon2, "should return same daemon instance")

    def test_multiple_interfaces_share_daemon(self):
        """test that multiple interfaces share the same daemon."""
        gnubg1 = GnubgInterface(plies=0)
        gnubg2 = GnubgInterface(plies=0)

        # both should work without starting separate daemons
        board = Board(standard_starting_points())
        hints1 = gnubg1.get_hint(board, Color.White, (3, 1))
        hints2 = gnubg2.get_hint(board, Color.Black, (4, 2))

        self.assertGreater(len(hints1), 0)
        self.assertGreater(len(hints2), 0)


class TestGnubgHints(unittest.TestCase):
    """test that gnubg returns reasonable hints for standard positions."""

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.board = Board(standard_starting_points())

    def test_white_gets_hints_from_starting_position(self):
        """gnubg should return at least one hint for white from starting position."""
        hints = self.gnubg.get_hint(self.board, Color.White, (3, 1))
        self.assertGreater(len(hints), 0, "gnubg should return hints for white")

    def test_black_gets_hints_from_starting_position(self):
        """gnubg should return at least one hint for black from starting position."""
        hints = self.gnubg.get_hint(self.board, Color.Black, (3, 1))
        self.assertGreater(len(hints), 0, "gnubg should return hints for black")

    def test_hints_return_move_strings(self):
        """gnubg hints should return move strings (socket interface doesn't return equity)."""
        hints = self.gnubg.get_hint(self.board, Color.White, (3, 1))
        self.assertGreater(len(hints), 0)
        # socket interface returns move strings, not equity
        self.assertIsNotNone(hints[0].moves)
        self.assertIsInstance(hints[0].moves, str)


class TestGnubgMoveNotationParsing(unittest.TestCase):
    """test parsing of gnubg move notation to our move format."""

    def setUp(self):
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))

        self.black_player = GnubgPlayer(Color.Black, plies=0)
        self.black_player.start_game(Game([BozoPlayer(Color.White), self.black_player]))

    def tearDown(self):
        pass

    def test_white_simple_move_parsing(self):
        """parse a simple move for white: 24/21"""
        # gnubg "24/21" for white = our point 1 moving distance 3
        moves = self.white_player._parse_gnubg_move_string("24/21", (3, 1))
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].color, Color.White)
        self.assertEqual(moves[0].point_number, 1)
        self.assertEqual(moves[0].distance, 3)

    def test_white_compound_move_parsing(self):
        """parse compound move for white: 13/10 6/5"""
        # gnubg "13/10" = our point 12 moving distance 3
        # gnubg "6/5" = our point 19 moving distance 1
        moves = self.white_player._parse_gnubg_move_string("13/10 6/5", (3, 1))
        self.assertEqual(len(moves), 2)

        # moves can be in either order
        move_set = set(moves)
        self.assertIn(Move(Color.White, 12, 3), move_set)
        self.assertIn(Move(Color.White, 19, 1), move_set)

    def test_black_simple_move_parsing(self):
        """parse a simple move for black: 1/4"""
        # for black, gnubg numbering matches ours
        moves = self.black_player._parse_gnubg_move_string("1/4", (3, 1))
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].color, Color.Black)
        self.assertEqual(moves[0].point_number, 1)
        self.assertEqual(moves[0].distance, 3)

    def test_black_compound_move_parsing(self):
        """parse compound move for black: 13/10 6/5"""
        moves = self.black_player._parse_gnubg_move_string("13/10 6/5", (3, 1))
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


class TestGnubgStrengthLevels(unittest.TestCase):
    """test that different gnubg strength levels work."""

    def test_gnubg_accepts_different_ply_settings(self):
        """gnubg should accept different ply settings from 0 to 4.

        note: 7-ply is world class but very slow, excluded from quick tests
        """
        for plies in [0, 1, 2, 4]:
            gnubg = GnubgInterface(plies=plies)

            board = Board(standard_starting_points())
            hints = gnubg.get_hint(board, Color.White, (3, 1))

            self.assertGreater(len(hints), 0, f"gnubg with {plies} plies should return hints")


class TestGnubgMoveMatching(unittest.TestCase):
    """test that gnubg suggestions match our possible_moves output.

    this is the critical integration test: if gnubg suggests moves that aren't
    in our possible_moves set, the player falls back to random moves.
    """

    def setUp(self):
        self.gnubg = GnubgInterface(plies=0)
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))
        self.black_player = GnubgPlayer(Color.Black, plies=0)
        self.black_player.start_game(Game([BozoPlayer(Color.White), self.black_player]))

    def tearDown(self):
        pass

    def test_white_opening_move_is_in_possible_moves(self):
        """gnubg's top suggestion for white opening should be in our possible_moves."""
        board = Board(standard_starting_points())
        dice_roll = (3, 1)

        # get gnubg's suggestion
        hints = self.gnubg.get_hint(board, Color.White, dice_roll)
        self.assertGreater(len(hints), 0, "gnubg should return hints")

        # parse it to our format
        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves, dice_roll)

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

        our_moves = self.black_player._parse_gnubg_move_string(hints[0].moves, dice_roll)

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

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves, dice_roll)

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
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))

    def tearDown(self):
        pass

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

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves, dice_roll)

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

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves, dice_roll)

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

        our_moves = self.white_player._parse_gnubg_move_string(hints[0].moves, dice_roll)

        dice = Dice(roll=dice_roll)
        possible = board.possible_moves(Color.White, dice)

        found = any(set(moves) == set(our_moves) for moves, _ in possible)
        self.assertTrue(found,
            f"gnubg suggested '{hints[0].moves}' for racing position "
            f"but it's not in our possible moves")




class TestGnubgBearingOffParsing(unittest.TestCase):
    """test bearing off move parsing bugs.

    these tests expose the bug where bearing off distance is calculated wrong.
    """

    def setUp(self):
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))
        self.black_player = GnubgPlayer(Color.Black, plies=0)
        self.black_player.start_game(Game([BozoPlayer(Color.White), self.black_player]))

    def tearDown(self):
        pass

    def test_white_bearing_off_simple(self):
        """white bearing off: "6/off" should mean bear off from gnubg point 6 using the die."""
        # gnubg point 6 = our point 19 for white
        # if rolling a 6, distance should be 6 (the die value)
        # NOT 19 (the point number)

        moves = self.white_player._parse_gnubg_move_string("6/off", (6, 1))

        self.assertEqual(len(moves), 1, "should parse exactly one move")
        move = moves[0]

        self.assertEqual(move.color, Color.White)
        self.assertEqual(move.point_number, 19,
            f"gnubg point 6 = our point 19, got {move.point_number}")

        # the bug: we calculate distance as our_source (19 or 20)
        # but it should be the die value used for bearing off
        # for point 19, minimum distance to bear off is 6
        # actual distance depends on which die was rolled
        self.assertLessEqual(move.distance, 6,
            f"bearing off from point 19 needs distance 6 at most, "
            f"but parsed distance is {move.distance}. "
            f"this suggests we're using the point number instead of die value.")

    def test_white_bearing_off_with_move(self):
        """compound move with bearing off: "5/2 5/off" from dice (6,3)."""
        # gnubg suggests "5/2 5/off" for dice (6,3)
        # "5/2" = gnubg point 5 to 2 = our point 20 to 23 = distance 3
        # "5/off" = bearing off from gnubg point 5 = our point 20 with die 6

        moves = self.white_player._parse_gnubg_move_string("5/2 5/off", (6, 3))

        self.assertEqual(len(moves), 2, "should parse exactly two moves")

        # find the bearing off move
        bearing_off_move = None
        regular_move = None
        for m in moves:
            if m.destination_is_off:
                bearing_off_move = m
            else:
                regular_move = m

        self.assertIsNotNone(bearing_off_move, "should have a bearing off move")
        self.assertIsNotNone(regular_move, "should have a regular move")

        # regular move should be correct
        self.assertEqual(regular_move.point_number, 20)
        self.assertEqual(regular_move.distance, 3)

        # bearing off move is the problem
        self.assertEqual(bearing_off_move.point_number, 20,
            "bearing off from gnubg point 5 = our point 20")

        # with dice (6,3), and "5/2" using the 3,
        # "5/off" must use the 6
        self.assertEqual(bearing_off_move.distance, 6,
            f"bearing off should use die value 6, "
            f"but parsed distance is {bearing_off_move.distance}")

    def test_black_bearing_off_simple(self):
        """black bearing off: "6/off" should use die value not point number."""
        # for black, gnubg point 6 = our point 6
        # bearing off from point 6 needs distance >= 6

        moves = self.black_player._parse_gnubg_move_string("6/off", (6, 1))

        self.assertEqual(len(moves), 1)
        move = moves[0]

        self.assertEqual(move.color, Color.Black)
        self.assertEqual(move.point_number, 6)

        # the bug applies to black too
        self.assertLessEqual(move.distance, 6,
            f"bearing off from point 6 with a 6 die should have distance 6, "
            f"but got {move.distance}")


class TestGnubgDoublesNotation(unittest.TestCase):
    """test doubles notation parsing bugs.

    these tests expose the bug where "(2)" repetition markers are ignored.
    """

    def setUp(self):
        self.white_player = GnubgPlayer(Color.White, plies=0)
        self.white_player.start_game(Game([self.white_player, BozoPlayer(Color.Black)]))
        self.black_player = GnubgPlayer(Color.Black, plies=0)
        self.black_player.start_game(Game([BozoPlayer(Color.White), self.black_player]))

    def tearDown(self):
        pass

    def test_doubles_notation_with_count(self):
        """gnubg uses "24/18(2)" to mean do 24/18 twice."""
        moves = self.white_player._parse_gnubg_move_string("24/18(2)", (6, 6))

        # for doubles, we need to expand "(2)" into two separate moves
        # but the bug is that we ignore "(2)" and only create one move
        self.assertEqual(len(moves), 2,
            f"'24/18(2)' means do the move twice, "
            f"but we parsed only {len(moves)} move(s): {moves}")

    def test_doubles_notation_multiple_parts(self):
        """complex doubles: "bar/19(2) 16/10(2)" means 4 total moves."""
        moves = self.white_player._parse_gnubg_move_string("bar/19(2) 16/10(2)", (6, 6))

        # "bar/19" twice + "16/10" twice = 4 moves total
        self.assertEqual(len(moves), 4,
            f"'bar/19(2) 16/10(2)' should expand to 4 moves, "
            f"but we parsed {len(moves)} move(s): {moves}")

        # verify the moves are correct
        bar_moves = [m for m in moves if m.point_number == Move.Bar]
        regular_moves = [m for m in moves if m.point_number != Move.Bar]

        self.assertEqual(len(bar_moves), 2,
            f"should have 2 bar entry moves, got {len(bar_moves)}")
        self.assertEqual(len(regular_moves), 2,
            f"should have 2 regular moves, got {len(regular_moves)}")

    def test_doubles_without_count_marker(self):
        """some doubles are written without (N): "13/7 13/7" instead of "13/7(2)"."""
        # gnubg sometimes writes doubles explicitly rather than with (2)
        moves = self.white_player._parse_gnubg_move_string("13/7 13/7", (6, 6))

        self.assertEqual(len(moves), 2,
            "'13/7 13/7' should parse as 2 separate moves")

        # both should be the same move
        self.assertEqual(moves[0], moves[1],
            "repeated notation should create identical moves")

    def test_mixed_doubles_notation(self):
        """mixed notation: "24/18(2) 18/16 18/16" = 4 moves."""
        moves = self.white_player._parse_gnubg_move_string("24/18(2) 18/16 18/16", (2, 2))

        self.assertEqual(len(moves), 4,
            f"should expand to 4 moves total, got {len(moves)}")


if __name__ == "__main__":
    unittest.main()
