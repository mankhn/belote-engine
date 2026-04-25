import random
from sre_parse import State
from typing import List, Tuple, Optional
from src.types import Action
from ..types import Agent, State, Log

class RandomChooser(Agent):
    def choose_action(self, _: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        return random.choice(actions), None
