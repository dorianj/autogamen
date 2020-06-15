from autogamen.ai.bozo import BozoPlayer
from autogamen.game.match import Match
from autogamen.game.types import Color

import sys

import random
random.seed(1234)

match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)], 25)

match.start()
while True:
  if match.tick():
    sys.exit()
