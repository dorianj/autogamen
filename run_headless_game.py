from autogamen.ai.bozo import BozoPlayer
from autogamen.game.match import Match
from autogamen.game.types import Color

import random
random.seed(1234)

match = Match([BozoPlayer(Color.White), BozoPlayer(Color.Black)])

match.start()
while True:
  match.tick()
