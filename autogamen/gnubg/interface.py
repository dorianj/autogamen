"""socket-based interface to gnubg for move evaluation and suggestions.

completely replaces the broken stdin/stdout implementation.
"""

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogamen.game.board import _Board
    from autogamen.game.game_types import Color


@dataclass
class GnubgMove:
    """a move suggestion from gnubg with its evaluation."""
    moves: str  # e.g. "24/18 13/11"
    equity: float = 0.0  # external interface doesn't return equity directly
    win_prob: float = 0.0
    win_gammon_prob: float = 0.0
    win_bg_prob: float = 0.0
    lose_prob: float = 0.0
    lose_gammon_prob: float = 0.0
    lose_bg_prob: float = 0.0


class GnubgDaemon:
    """manages the gnubg daemon process with external interface."""

    def __init__(self, port: int = 12345):
        self.port = port
        self.process: subprocess.Popen[bytes] | None = None
        self.gnubg_path = str(Path(__file__).parent.parent.parent / "vendor" / "gnubg" / "gnubg")
        self.data_dir = str(Path(__file__).parent.parent.parent / "vendor" / "gnubg")

    def __del__(self) -> None:
        self.stop()

    def start(self) -> None:
        """start gnubg daemon listening on specified port."""
        if self.process is not None:
            # check if process is still alive
            if self.process.poll() is None:
                return  # already running
            else:
                self.process = None  # process died, need to restart

        # kill any existing gnubg processes that might be blocking the port
        subprocess.run(["pkill", "-f", f"gnubg.*external.*{self.port}"], capture_output=True)
        time.sleep(0.2)

        cmd = [
            self.gnubg_path,
            "-t",  # tty mode
            "-q",  # quiet
            f"--pkgdatadir={self.data_dir}",
            f"--datadir={self.data_dir}",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # wait for gnubg to start
        time.sleep(0.5)

        # send external command to start socket listener
        if self.process.stdin is not None:
            self.process.stdin.write(f"external localhost:{self.port}\n".encode())
            self.process.stdin.flush()

        # wait for socket to be ready
        time.sleep(1.0)

    def stop(self) -> None:
        """stop the gnubg daemon."""
        if self.process is None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


# global daemon instance - single gnubg process for all games
_daemon: GnubgDaemon | None = None


def get_daemon(port: int = 12345) -> GnubgDaemon:
    """get or create the global gnubg daemon."""
    global _daemon
    if _daemon is None:
        _daemon = GnubgDaemon(port)
        _daemon.start()
    return _daemon


class GnubgInterface:
    """socket-based interface to gnubg external controller."""

    def __init__(self, plies: int = 2):
        self.plies = plies
        self.port = 12345
        self.host = 'localhost'

        # ensure daemon is running
        get_daemon(self.port)

    def _board_to_fibs(self, board: "_Board", player_color: "Color", dice: tuple[int, int]) -> str:
        """convert our board to FIBS board format.

        working example that gnubg accepts and responds to:
        board:You:opponent:1:0:0:0:-2:0:0:0:0:5:0:3:0:0:0:-5:5:0:0:0:-3:0:-5:0:0:0:0:2:0:1:6:3:0:0:1:1:1:0:1:-1:0:25:0:0:0:0:2:0:0:0

        format: board:p1:p2:match:s1:s2:[26 board positions]:turn:d1:d2:d3:d4:cube:dbl1:dbl2:was_dbl:color:dir:[trailing fields]
        """
        from autogamen.game.game_types import Color  # noqa: PLC0415

        # build the 26-element board array for FIBS
        # format: [opponent_bar, points 1-24 from player perspective, player_bar]
        # positive = player (X), negative = opponent (O)

        board_values = []

        # opponent bar (negative because it's opponent's pieces)
        opponent_bar = board.bar[player_color.opponent().value]
        board_values.append(-opponent_bar if opponent_bar else 0)

        # points 1-24 from player's perspective
        for fibs_point in range(1, 25):
            if player_color == Color.White:
                # white perspective: FIBS point 1 = our point 24, FIBS 24 = our point 1
                our_point = 25 - fibs_point
            else:
                # black perspective: FIBS point 1 = our point 1, FIBS 24 = our point 24
                our_point = fibs_point

            point = board.point_at_number(our_point)
            if point.is_empty():
                board_values.append(0)
            elif point.color == player_color:
                board_values.append(point.count)  # positive for player
            else:
                board_values.append(-point.count)  # negative for opponent

        # player bar
        player_bar = board.bar[player_color.value]
        board_values.append(player_bar)

        # construct FIBS string using the working format
        parts = [
            "board",
            "You",
            "opponent",
            "1",  # match length
            "0",  # score 1
            "0",  # score 2
        ]

        # add each board position as separate field
        parts.extend(str(v) for v in board_values)

        # add game state fields
        parts.append("1")  # turn (1=player)

        # dice fields - for doubles, all 4 should have the same value
        if dice[0] == dice[1]:
            # doubles
            parts.extend([str(dice[0]), str(dice[0]), str(dice[0]), str(dice[0])])
        else:
            # normal roll
            parts.extend([str(dice[0]), str(dice[1]), "0", "0"])

        parts.extend([
            "1",  # cube
            "1",  # may double 1
            "1",  # may double 2
            "0",  # was doubled
            "1" if player_color == Color.White else "-1",  # color
            "-1" if player_color == Color.White else "1",  # direction
            "0", "25", "0", "0", "0", "0", "2", "0", "0", "0"  # trailing fields from working example
        ])

        return ":".join(parts)

    def get_hint(self, board: "_Board", color: "Color", dice: tuple[int, int]) -> list[GnubgMove]:
        """get move suggestions from gnubg for the given position."""
        daemon = get_daemon(self.port)

        # gnubg's external interface closes the socket listener after each request
        # we need to restart it before every get_hint() call
        if daemon.process and daemon.process.stdin and daemon.process.poll() is None:
            daemon.process.stdin.write(f"external localhost:{self.port}\n".encode())
            daemon.process.stdin.flush()
            time.sleep(0.1)

        # create new socket connection for this request
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # connect with retries
            for attempt in range(5):
                try:
                    sock.connect((self.host, self.port))
                    break
                except ConnectionRefusedError:
                    if attempt < 4:
                        time.sleep(0.2)
                    else:
                        return []  # couldn't connect

            # convert board to FIBS format
            fibs_board = self._board_to_fibs(board, color, dice)

            # send board to gnubg
            sock.sendall(fibs_board.encode() + b"\n")

            # receive response
            response = sock.recv(4096).decode().strip()

            # parse response
            if not response:
                return []

            if response.startswith("Error"):
                return []

            # gnubg returns the best move directly
            return [GnubgMove(moves=response)]

        except Exception:
            # connection error or gnubg not running
            return []
        finally:
            sock.close()
