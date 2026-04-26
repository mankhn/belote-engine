from typing import List, Tuple, Optional
from src.kits import CardKit, ListKit
from ..types import Action, State, Log
from ..core.agent import Agent
from ..kits import ActionKit


class SoftPlayer(Agent):
    """Soft player agent that tries to lose the trick"""

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        """Choose the card that loses the trick, or lowest value card"""
        play_actions = [a for a in actions if ActionKit.is_play_card(a)]

        if not play_actions:
            return actions[0] if actions else None, None

        trump = state.trump

        def card_value(a):
            return CardKit.value(ActionKit.value(a), trump)

        # If leading (table empty), play lowest value card
        if not state.table:
            return min(play_actions, key=card_value), None

        # Find current winner on table
        current_winner, _ = ListKit.winner(state.table, trump)

        # Find cards that DO NOT beat the current winner
        losing_actions = [a for a in play_actions if not CardKit.beats(ActionKit.value(a), trump, current_winner)]

        if losing_actions:
            return min(losing_actions, key=card_value), None
        return min(play_actions, key=card_value), None

