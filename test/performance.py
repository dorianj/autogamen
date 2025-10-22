import cProfile
import io
import pstats
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def assertRuntime(self: Any, expected_time: float) -> Generator[None, None, None]:  # noqa: N802
  pr = cProfile.Profile()
  pr.enable()
  start_time = time.perf_counter()
  try:
    yield None
  finally:
    pr.disable()
    actual_time = time.perf_counter() - start_time

    if actual_time > expected_time:
      s = io.StringIO()
      ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
      ps.print_stats(0.2)
      print(s.getvalue())

    self.assertLessEqual(actual_time, expected_time, msg="Too slow")
