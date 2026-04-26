import random
from typing import List, Tuple, Optional
from ..types import Action, State, Log
from ..core.agent import Agent


class RandomChooser(Agent):
    def choose_action(self, _: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        return random.choice(actions), None

