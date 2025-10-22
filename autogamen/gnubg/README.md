# gnubg integration

this module provides integration with GNU Backgammon (gnubg), a world-class backgammon AI.

## overview

we've vendored gnubg source code and built it as part of our project. the integration uses subprocess communication via stdin/stdout to:
- send board positions to gnubg
- request move evaluations and suggestions
- parse gnubg's analysis output

## components

### `interface.py`
core gnubg subprocess wrapper:
- `GnubgInterface`: manages gnubg process lifecycle
- `GnubgMove`: dataclass representing a move suggestion with equity and probabilities
- board conversion between our format and gnubg's "simple" format
- move hint parsing from gnubg output

### `../ai/players.py`
contains `GnubgPlayer` class that uses gnubg to make move decisions

## usage

### using GnubgPlayer in a game

```python
from autogamen.ai.players import GnubgPlayer, BozoPlayer
from autogamen.game.game import Game
from autogamen.game.game_types import Color

white = GnubgPlayer(Color.White)
black = BozoPlayer(Color.Black)
game = Game([white, black])
game.start()

while not game.winner:
    game.run_turn()
```

### using GnubgInterface directly

```python
from autogamen.gnubg.interface import GnubgInterface
from autogamen.game.board import Board
from autogamen.game.game_types import Color

with GnubgInterface() as gnubg:
    hints = gnubg.get_hint(board, Color.White, (3, 5))

    for hint in hints[:5]:  # top 5 suggestions
        print(f"{hint.moves:30s} eq: {hint.equity:+.3f}")
```

## implementation details

### IPC mechanism
we use stdin/stdout pipes with the gnubg subprocess. gnubg runs in tty mode (`-t` flag) and accepts text commands.

### board format conversion
gnubg uses a "simple" format: 24 integers representing points 1-24, plus bar count:
- positive numbers = player on roll (X)
- negative numbers = opponent (O)
- 0 = empty point

our conversion function (`_board_to_gnubg_simple`) handles mapping our board representation to this format, accounting for the fact that gnubg always treats the current player as X.

### reading gnubg output
since gnubg doesn't provide a clear prompt in tty mode, we use a timeout-based approach with `select()` to detect when gnubg is waiting for input.

### move notation
gnubg uses notation like "24/18 13/11" (from/to format). we parse this and convert to our `Move` objects.

## testing

run the integration tests:
```bash
uv run python test_gnubg.py           # basic hint test
uv run python test_gnubg_game.py      # full game test
uv run python run_gnubg.py            # interactive gnubg shell
```

## trade-offs

**current approach (stdin/stdout):**
- ✔ simple implementation
- ✔ easy to debug (can see commands/output)
- ✘ text parsing can be brittle
- ✘ slight overhead from text conversion

**alternative (FIBS socket protocol):**
- ✔ structured binary protocol
- ✔ more robust
- ✘ more complex to implement
- ✘ requires understanding FIBS board format

we chose stdin/stdout for simplicity. can migrate to FIBS protocol later if needed.

## building gnubg

gnubg is built via makefile:
```bash
make gnubg  # builds vendor/gnubg/gnubg binary
```

the binary needs data files (`.bd`, `.wd`, `.weights`) which are passed via `--pkgdatadir` and `--datadir` flags.
