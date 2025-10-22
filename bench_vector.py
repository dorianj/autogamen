#!/usr/bin/env python3
"""benchmark vector_game operations

usage:
  python bench_vector.py           # timing only
  PROFILE=1 python bench_vector.py # with profiling
"""
import cProfile
import io
import os
import pstats
import time

from autogamen.game import vector_game as vg
from autogamen.game.board import Board
from autogamen.game.game_types import Color, Move, Point


def setup_board():
    return [
        Point(2, Color.White), Point(), Point(), Point(), Point(),
        Point(5, Color.Black), Point(0), Point(3, Color.Black),
        Point(), Point(), Point(), Point(5, Color.White),
        Point(5, Color.Black), Point(), Point(), Point(),
        Point(3, Color.White), Point(0), Point(5, Color.White),
        Point(), Point(), Point(), Point(), Point(2, Color.Black),
    ]


def main():
    enable_profiling = os.environ.get('PROFILE') == '1'
    profiler = cProfile.Profile() if enable_profiling else None

    print('\n§ vector_game benchmarks\n')

    board = Board(setup_board())
    vec = vg.from_board(board)

    if not enable_profiling:
        # benchmark 1: apply_move
        move = Move(Color.White, 12, 3)
        iterations = 45000

        start = time.perf_counter()
        for _ in range(iterations):
            _ = vg.apply_move(vec, move)
        elapsed = time.perf_counter() - start

        print(f'✱ apply_move: {elapsed*1000:.2f}ms ({iterations} iters, {elapsed/iterations*1000000:.2f}µs/call)')

        # benchmark 2: possible_moves
        dice = (3, 5)
        iterations = 3000

        start = time.perf_counter()
        for _ in range(iterations):
            _ = vg.possible_moves(vec, Color.White, dice)
        elapsed = time.perf_counter() - start

        print(f'✱ possible_moves: {elapsed*1000:.2f}ms ({iterations} iters, {elapsed/iterations*1000:.2f}ms/call)')

    # benchmark 3: RL training simulation (most realistic workload)
    dice = (4, 2)
    iterations = 2000

    if profiler:
        profiler.enable()
    start = time.perf_counter()
    for _ in range(iterations):
        moves = vg.possible_moves(vec, Color.White, dice)
        for _, result_vec in moves:
            _ = result_vec
    elapsed = time.perf_counter() - start
    if profiler:
        profiler.disable()

    print(f'✱ RL simulation: {elapsed*1000:.2f}ms ({iterations} iters, {elapsed/iterations*1000:.2f}ms/call)')

    # print profile if enabled
    if enable_profiling and profiler:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(20)
        print('\n\n§ profile (top 20 by cumulative time):\n')
        print(s.getvalue())


if __name__ == '__main__':
    main()
