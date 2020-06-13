from contextlib import contextmanager
import time

@contextmanager
def assertRuntime(self, expected_time):
  start_time = time.perf_counter()
  try:
    yield None
  finally:
    actual_time = time.perf_counter() - start_time
    self.assertLessEqual(actual_time, expected_time, msg="Too slow")
