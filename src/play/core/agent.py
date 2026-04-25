from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..kits import ObjectiveKit
from ..types import Objective, State, Action, Log

class Agent(ABC):

    def __init__(self, objective: Objective = ObjectiveKit.max_score()):
        self.objective = objective

    @abstractmethod
    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        """
        Choose an action based on the current game state.
        
        Args:
            state (State): The current state of the game.
            actions (List[Action]): The list of possible actions.
            objective (Objective): The objective the agent is trying to achieve.

        Returns:
            Tuple[Action, Optional[Log]]: A tuple containing the action chosen by the agent and a log that can be used for training.
        """
        pass
