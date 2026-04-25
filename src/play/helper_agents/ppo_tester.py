from typing import Any, Dict, List, Tuple, Optional

from ..types import State, Action, Record
from ..kits import ActionKit
from ..ppo.agent import PpoAgent


class PpoTester(PpoAgent):
    # static variables for all instances
    cursor = 0
    total_moves = 0
    total_matches = 0

    def __init__(self, *args, records: List[Record]):
        super().__init__(*args)
        PpoTester.records = records

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Dict[str, Any]]]:
        record = PpoTester.records[PpoTester.cursor]
        # Record is a tuple: (player, state, action, reward, log)
        recorded_action = record[2]
        PpoTester.cursor += 1

        agent_action, _ = super().choose_action(state, actions)

        if not ActionKit.is_pass(agent_action):
            if agent_action == recorded_action:
                PpoTester.total_matches += 1
            PpoTester.total_moves += 1

        return recorded_action, None

        
