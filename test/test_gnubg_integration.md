# gnubg integration test results

## bug found and fixed

**the bug**: board conversion to gnubg format was using incorrect point numbering for white.

gnubg uses player-relative point numbering:
- for white: gnubg point 24 = starting position (our point 1), gnubg point 1 = bear-off area (our point 24)
- for black: gnubg point numbering matches ours directly

the old code was iterating through our points 1-24 and sending them as gnubg points 1-24, which meant white was seeing a completely backwards board. this caused gnubg to either fail to return hints or return hints for an invalid position.

**the fix**: in `autogamen/gnubg/interface.py`, the `_board_to_gnubg_simple()` method now converts point numbers:
```python
for gnubg_point_num in range(1, 25):
    if player_color == Color.White:
        our_point_num = 25 - gnubg_point_num
    else:
        our_point_num = gnubg_point_num
    # ... rest of conversion
```

similarly, in `autogamen/ai/players.py`, the `_parse_gnubg_move_string()` method now uses `_gnubg_to_our_point()` to convert point numbers when parsing gnubg's move notation.

## tests that pass

all 12 integration tests pass:

1. **board conversion tests** - verify correct point numbering for both white and black
2. **hint retrieval tests** - gnubg returns valid hints for both colors
3. **move parsing tests** - gnubg notation is correctly parsed to our move format
4. **gameplay tests** - gnubg can complete full games vs random and vs itself
5. **strength level tests** - different ply settings (0-7) work correctly

## known issue: move matching failures

the tests reveal that gnubg's suggested moves often don't match any move in our `possible_moves` set. examples from test output:

- `"8/5 6/5"` - both moves landing on same point
- `"24/13"` - single die moving full distance
- `"8/7(2) 6/5(2)"` - doubles notation with (2)
- `"8/4*"` - hitting notation with *
- moves using two dice from same point

**why this happens**: our move generator may not be generating all legal moves, or gnubg's notation includes moves we don't support. this needs investigation.

**impact**: when no match is found, gnubg player falls back to random move selection. this means gnubg is currently playing mostly random moves during actual games, despite the integration being "correct" in terms of board representation and hint parsing.

## what's working

- ✅ board state is correctly sent to gnubg
- ✅ gnubg returns valid hints with proper equity values
- ✅ move notation parsing converts gnubg format to our format
- ✅ point number conversion is correct for both colors
- ✅ games can complete without errors

## what needs fixing

- ❌ move matching: gnubg's suggested moves don't match our possible moves
  - could be missing move combinations in our move generator
  - could be notation we don't parse (like `(2)` for doubles, `*` for hits)
  - could be legal moves we don't generate

## how the test suite helps

this test suite would have caught the board conversion bug immediately:
- `test_white_board_conversion` checks specific points match expected values
- `test_black_board_conversion` verifies black uses different numbering

it also documents expected behavior:
- how point numbering differs between white and black
- what gnubg notation looks like and how it should parse
- that gnubg can play complete games (even if not optimally)

## running the tests

```bash
uv run python -m unittest test.test_gnubg_integration -v
```

all tests should pass, but you may see resource warnings about unclosed file descriptors (known gnubg subprocess cleanup issue).
