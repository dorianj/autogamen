"""interface to gnubg for move evaluation and suggestions."""
import os
import re
import select
import subprocess
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
    equity: float
    win_prob: float
    win_gammon_prob: float
    win_bg_prob: float
    lose_prob: float
    lose_gammon_prob: float
    lose_bg_prob: float


class GnubgInterface:
    """manages communication with gnubg subprocess."""

    def __init__(self, gnubg_path: str | None = None, data_dir: str | None = None, plies: int = 2):
        # locate gnubg binary and data directory
        if gnubg_path is None:
            repo_root = Path(__file__).parent.parent.parent
            gnubg_path = str(repo_root / "vendor" / "gnubg" / "gnubg")

        if data_dir is None:
            repo_root = Path(__file__).parent.parent.parent
            data_dir = str(repo_root / "vendor" / "gnubg")

        if not os.path.exists(gnubg_path):
            raise FileNotFoundError(f"gnubg binary not found at {gnubg_path}")

        self.gnubg_path = gnubg_path
        self.data_dir = data_dir
        self.plies = plies
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """start the gnubg subprocess."""
        if self.process is not None:
            raise RuntimeError("gnubg already started")

        cmd = [
            self.gnubg_path,
            "-t",  # tty mode
            "-q",  # quiet (no sound)
            f"--pkgdatadir={self.data_dir}",
            f"--datadir={self.data_dir}",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # consume the startup banner
        self._read_until_prompt()

        # configure evaluation strength
        self._send_command(f"set evaluation chequerplay evaluation plies {self.plies}")
        self._read_until_prompt()
        self._send_command(f"set evaluation cubedecision evaluation plies {self.plies}")
        self._read_until_prompt()

    def stop(self) -> None:
        """stop the gnubg subprocess."""
        if self.process is None:
            return

        try:
            self._send_command("quit")
            self.process.wait(timeout=5)
        except Exception:
            # force kill if quit doesn't work
            self.process.kill()
        finally:
            self.process = None

    def _send_command(self, command: str) -> None:
        """send a command to gnubg."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("gnubg not started")

        self.process.stdin.write(f"{command}\n".encode())
        self.process.stdin.flush()

    def _read_until_prompt(self, debug: bool = False) -> str:
        """read output until we see a prompt (empty line after output).

        gnubg doesn't have a clear prompt in tty mode, so we use a timeout-based
        approach: read lines until we get a timeout, which indicates gnubg is
        waiting for input.
        """
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("gnubg not started")

        output_lines = []
        timeout = 0.1  # wait up to 100ms for more output

        while True:
            # use select to check if data is available
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)

            if not ready:
                # timeout - gnubg is probably waiting for input
                break

            line = self.process.stdout.readline().decode().rstrip("\n\r")
            if debug:
                print(f"[gnubg] {line}")

            output_lines.append(line)

        return "\n".join(output_lines)

    def _board_to_gnubg_simple(self, board: "_Board", player_color: "Color") -> str:
        """convert our board to gnubg simple format.

        gnubg simple format: 26 integers total
        - integer 1: player's bar count (non-negative)
        - integers 2-25: points 1-24 (positive = player on roll, negative = opponent)
        - integer 26: opponent's bar count (non-negative)

        gnubg uses player-relative point numbering:
        - for white: gnubg point 1 = our point 24, gnubg point 24 = our point 1
        - for black: gnubg point 1 = our point 1, gnubg point 24 = our point 24
        """
        from autogamen.game.game_types import Color  # noqa: PLC0415

        # build the 24-element array for gnubg
        gnubg_points = []

        for gnubg_point_num in range(1, 25):
            # convert gnubg point number to our point number
            if player_color == Color.White:
                our_point_num = 25 - gnubg_point_num
            else:
                our_point_num = gnubg_point_num

            point = board.point_at_number(our_point_num)

            if point.is_empty():
                gnubg_points.append(0)
            else:
                # in gnubg, X (positive) is the player on roll
                sign = 1 if point.color == player_color else -1
                gnubg_points.append(sign * point.count)

        player_bar_count = board.bar[player_color.value]
        opponent_bar_count = board.bar[player_color.opponent().value]

        # correct format: player_bar, then 24 points, then opponent_bar
        points_str = " ".join(str(p) for p in gnubg_points)
        return f"simple {player_bar_count} {points_str} {opponent_bar_count}"

    def get_hint(self, board: "_Board", color: "Color", dice: tuple[int, int]) -> list[GnubgMove]:
        """get move suggestions from gnubg for the given position."""
        # start a new game
        self._send_command("new game")
        self._read_until_prompt()

        # set up the board position
        board_cmd = f"set board {self._board_to_gnubg_simple(board, color)}"
        self._send_command(board_cmd)
        self._read_until_prompt()

        # set the dice for this position
        self._send_command(f"set dice {dice[0]} {dice[1]}")
        self._read_until_prompt()

        # ask for hint
        self._send_command("hint")
        output = self._read_until_prompt()

        # parse the hint output
        return self._parse_hint_output(output)

    def _parse_hint_output(self, output: str) -> list[GnubgMove]:
        """parse gnubg hint output to extract move suggestions.

        gnubg output looks like:
            1. Cubeful 2-ply    24/18 13/11                  Eq.:  +0.017
               0.508 0.131 0.006 - 0.492 0.136 0.006
                2-ply cubeful prune [world class]
        """
        moves = []
        lines = output.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # match move line: "1. Cubeful 2-ply    24/18 13/11    Eq.:  +0.017"
            move_match = re.match(
                r'\d+\.\s+(?:Cubeful|Cubeless)\s+\S+\s+(.*?)\s+Eq\.:\s+([-+]?\d+\.\d+)',
                line
            )

            if move_match:
                move_str = move_match.group(1).strip()
                equity = float(move_match.group(2))

                # next line should have probabilities
                if i + 1 < len(lines):
                    prob_line = lines[i + 1].strip()
                    # "0.508 0.131 0.006 - 0.492 0.136 0.006"
                    prob_match = re.match(
                        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                        prob_line
                    )

                    if prob_match:
                        moves.append(GnubgMove(
                            moves=move_str,
                            equity=equity,
                            win_prob=float(prob_match.group(1)),
                            win_gammon_prob=float(prob_match.group(2)),
                            win_bg_prob=float(prob_match.group(3)),
                            lose_prob=float(prob_match.group(4)),
                            lose_gammon_prob=float(prob_match.group(5)),
                            lose_bg_prob=float(prob_match.group(6)),
                        ))

            i += 1

        return moves

    def __enter__(self) -> "GnubgInterface":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
