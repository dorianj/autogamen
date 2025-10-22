"""vector-native game implementation for high-throughput RL training.

stays in tensor representation throughout, avoiding expensive round-trips
through Board/Point objects. only converts at boundaries for testing/UI.
"""
import numpy as np
import numpy.typing as npt

from autogamen.game.board import Board
from autogamen.game.game_types import Color, Move, Point

# vector representation (198 elements):
# [0-191]: 24 points × 8 features per point:
#   - [0-3]: white features (≥1, ≥2, ≥3, (count-3)/2)
#   - [4-7]: black features (≥1, ≥2, ≥3, (count-3)/2)
# [192]: white bar count
# [193]: black bar count
# [194]: white off count
# [195]: black off count
# [196]: active color white (1 or 0)
# [197]: active color black (1 or 0)

VECTOR_SIZE = 198
POINTS_SECTION_SIZE = 192  # 24 points × 8 features
BAR_WHITE_IDX = 192
BAR_BLACK_IDX = 193
OFF_WHITE_IDX = 194
OFF_BLACK_IDX = 195
ACTIVE_WHITE_IDX = 196
ACTIVE_BLACK_IDX = 197


def from_board(board: Board, active_color: Color | None = None) -> npt.NDArray[np.float32]:
    """convert Board to vector representation"""
    vec = np.zeros(VECTOR_SIZE, dtype=np.float32)

    # encode 24 points
    for i, point in enumerate(board.points):
        base_idx = i * 8

        if point.color == Color.White:
            vec[base_idx + 0] = 1.0 if point.count >= 1 else 0.0
            vec[base_idx + 1] = 1.0 if point.count >= 2 else 0.0
            vec[base_idx + 2] = 1.0 if point.count >= 3 else 0.0
            vec[base_idx + 3] = (point.count - 3) / 2.0 if point.count > 3 else 0.0
        elif point.color == Color.Black:
            vec[base_idx + 4] = 1.0 if point.count >= 1 else 0.0
            vec[base_idx + 5] = 1.0 if point.count >= 2 else 0.0
            vec[base_idx + 6] = 1.0 if point.count >= 3 else 0.0
            vec[base_idx + 7] = (point.count - 3) / 2.0 if point.count > 3 else 0.0

    # encode bar/off
    vec[BAR_WHITE_IDX] = float(board.bar[Color.White.value])
    vec[BAR_BLACK_IDX] = float(board.bar[Color.Black.value])
    vec[OFF_WHITE_IDX] = float(board.off[Color.White.value])
    vec[OFF_BLACK_IDX] = float(board.off[Color.Black.value])

    # encode active color if provided
    if active_color is not None:
        vec[ACTIVE_WHITE_IDX] = 1.0 if active_color == Color.White else 0.0
        vec[ACTIVE_BLACK_IDX] = 1.0 if active_color == Color.Black else 0.0

    return vec


def to_board(vec: npt.NDArray[np.float32]) -> Board:
    """convert vector back to Board (for testing/debugging)"""
    points = []
    for i in range(24):
        base_idx = i * 8

        # check white first
        if vec[base_idx + 0] > 0.5:  # has white checkers
            # reconstruct count from encoding
            if vec[base_idx + 2] > 0.5:  # ≥3
                count = 3 + int(vec[base_idx + 3] * 2)
            elif vec[base_idx + 1] > 0.5:  # ≥2
                count = 2
            else:  # ==1
                count = 1
            points.append(Point(count, Color.White))
        # check black
        elif vec[base_idx + 4] > 0.5:  # has black checkers
            if vec[base_idx + 6] > 0.5:  # ≥3
                count = 3 + int(vec[base_idx + 7] * 2)
            elif vec[base_idx + 5] > 0.5:  # ≥2
                count = 2
            else:  # ==1
                count = 1
            points.append(Point(count, Color.Black))
        else:
            points.append(Point(0, None))

    bar = [int(vec[BAR_WHITE_IDX]), int(vec[BAR_BLACK_IDX])]
    off = [int(vec[OFF_WHITE_IDX]), int(vec[OFF_BLACK_IDX])]

    return Board(points, bar, off)


def _get_point_color_count(vec: npt.NDArray[np.float32], point_num: int) -> tuple[Color | None, int]:
    """extract color and count for a point from vector"""
    base_idx = (point_num - 1) * 8

    # check white
    if vec[base_idx + 0] > 0.5:
        if vec[base_idx + 2] > 0.5:  # ≥3
            count = 3 + int(vec[base_idx + 3] * 2)
        elif vec[base_idx + 1] > 0.5:  # ≥2
            count = 2
        else:
            count = 1
        return Color.White, count

    # check black
    if vec[base_idx + 4] > 0.5:
        if vec[base_idx + 6] > 0.5:  # ≥3
            count = 3 + int(vec[base_idx + 7] * 2)
        elif vec[base_idx + 5] > 0.5:  # ≥2
            count = 2
        else:
            count = 1
        return Color.Black, count

    return None, 0


def _set_point(vec: npt.NDArray[np.float32], point_num: int, color: Color | None, count: int) -> None:
    """update a point in the vector"""
    base_idx = (point_num - 1) * 8

    # clear both colors first
    vec[base_idx:base_idx + 8] = 0.0

    if color is None or count == 0:
        return

    offset = 0 if color == Color.White else 4
    vec[base_idx + offset + 0] = 1.0 if count >= 1 else 0.0
    vec[base_idx + offset + 1] = 1.0 if count >= 2 else 0.0
    vec[base_idx + offset + 2] = 1.0 if count >= 3 else 0.0
    vec[base_idx + offset + 3] = (count - 3) / 2.0 if count > 3 else 0.0


def apply_move(vec: npt.NDArray[np.float32], move: Move) -> npt.NDArray[np.float32]:
    """apply a single move to vector, returning new vector"""
    result = vec.copy()

    # handle source: bar or point
    if move.point_number == Move.Bar:
        bar_idx = BAR_WHITE_IDX if move.color == Color.White else BAR_BLACK_IDX
        result[bar_idx] -= 1
    else:
        source_color, source_count = _get_point_color_count(result, move.point_number)
        if source_color != move.color or source_count == 0:
            raise ValueError(f"invalid move: no {move.color} checker at point {move.point_number}")
        _set_point(result, move.point_number, move.color if source_count > 1 else None, source_count - 1)

    # handle destination: off or point
    if move.destination_is_off:
        off_idx = OFF_WHITE_IDX if move.color == Color.White else OFF_BLACK_IDX
        result[off_idx] += 1
    else:
        assert move.destination_point_number is not None
        dest_color, dest_count = _get_point_color_count(result, move.destination_point_number)

        # check for hit
        if dest_color is not None and dest_color != move.color and dest_count == 1:
            # hit: remove opponent checker to bar, place our checker
            bar_idx = BAR_WHITE_IDX if dest_color == Color.White else BAR_BLACK_IDX
            result[bar_idx] += 1
            _set_point(result, move.destination_point_number, move.color, 1)
        else:
            # normal landing
            _set_point(result, move.destination_point_number, move.color, dest_count + 1)

    return result


def can_bear_off(vec: npt.NDArray[np.float32], color: Color) -> bool:
    """check if color can bear off from this position"""
    bar_idx = BAR_WHITE_IDX if color == Color.White else BAR_BLACK_IDX
    if vec[bar_idx] > 0.5:
        return False

    # home board range
    home_start, home_end = (19, 24) if color == Color.White else (1, 6)

    # check if any checkers outside home board
    for point_num in range(1, 25):
        if home_start <= point_num <= home_end:
            continue
        point_color, point_count = _get_point_color_count(vec, point_num)
        if point_color == color and point_count > 0:
            return False

    return True


def move_is_valid(vec: npt.NDArray[np.float32], move: Move) -> bool:
    """check if move is legal in current position"""
    bar_idx = BAR_WHITE_IDX if move.color == Color.White else BAR_BLACK_IDX

    # if coming from bar
    if move.point_number == Move.Bar:
        if vec[bar_idx] < 0.5:
            return False
        # check destination
        assert move.destination_point_number is not None
        dest_color, dest_count = _get_point_color_count(vec, move.destination_point_number)
        # can land if empty, or opponent singleton, or our color
        return dest_color is None or (dest_color != move.color and dest_count == 1) or dest_color == move.color

    # if on bar, can't move from board
    if vec[bar_idx] > 0.5:
        return False

    # check source point
    source_color, source_count = _get_point_color_count(vec, move.point_number)
    if source_color != move.color or source_count == 0:
        return False

    # if bearing off
    if move.destination_is_off:
        return can_bear_off(vec, move.color)

    # check destination point
    assert move.destination_point_number is not None
    dest_color, dest_count = _get_point_color_count(vec, move.destination_point_number)
    return dest_color is None or (dest_color != move.color and dest_count == 1) or dest_color == move.color


def possible_moves(vec: npt.NDArray[np.float32], color: Color, dice: tuple[int, ...]) -> list[tuple[tuple[Move, ...], npt.NDArray[np.float32]]]:
    """generate all legal move sequences for this position

    returns list of (move_sequence, resulting_board_vector)
    """
    def _worker(current_vec: npt.NDArray[np.float32], remaining_dice: tuple[int, ...]) -> list[tuple[tuple[Move, ...], npt.NDArray[np.float32]]]:
        moves: list[tuple[tuple[Move, ...], npt.NDArray[np.float32]]] = []

        bar_idx = BAR_WHITE_IDX if color == Color.White else BAR_BLACK_IDX
        on_bar = current_vec[bar_idx] > 0.5

        for d_idx, die in enumerate(set(remaining_dice)):
            # try all possible sources: bar (0) and points (1-24)
            for point_num in range(0, 25):
                # early rejection: skip moves from board points if we're on bar
                if on_bar and point_num != Move.Bar:
                    continue

                # early rejection: skip bar moves if not on bar
                if not on_bar and point_num == Move.Bar:
                    continue

                # early rejection: skip points without our checkers
                if point_num > 0:  # not bar
                    # inline point color check for hot path performance
                    base_idx = (point_num - 1) * 8
                    offset = 0 if color == Color.White else 4
                    if current_vec[base_idx + offset] < 0.5:  # no checkers of our color
                        continue

                move = Move(color, point_num, die)

                if move_is_valid(current_vec, move):
                    new_vec = apply_move(current_vec, move)
                    moves.append(((move,), new_vec))

                    # recurse for additional moves
                    new_remaining = remaining_dice[:d_idx] + remaining_dice[d_idx + 1:]
                    if new_remaining:
                        for submoves, subvec in _worker(new_vec, new_remaining):
                            moves.append(((move,) + submoves, subvec))

        return moves

    all_moves = _worker(vec, dice)

    if not all_moves:
        return []

    # prune: keep only moves that use maximum dice
    max_len = max(len(m[0]) for m in all_moves)
    return [m for m in all_moves if len(m[0]) == max_len]


def winner(vec: npt.NDArray[np.float32]) -> Color | None:
    """check if there's a winner"""
    if vec[OFF_WHITE_IDX] >= 14.5:  # == 15
        return Color.White
    if vec[OFF_BLACK_IDX] >= 14.5:
        return Color.Black
    return None
