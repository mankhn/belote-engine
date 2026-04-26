from typing import List, Tuple, Optional
from src.kits import CardKit, ListKit
from ..types import Action, State, Log
from ..core.agent import Agent
from ..kits import ActionKit


class AggressivePlayer(Agent):
    """Aggressive player agent that tries to win the trick"""

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        """Choose the card that wins the trick, or highest value card"""
        play_actions = [a for a in actions if ActionKit.is_play_card(a)]

        if not play_actions:
            return actions[0] if actions else None, None

        trump = state.trump

        def card_value(a):
            return CardKit.value(ActionKit.value(a), trump)

        # If leading (table empty), play highest value card
        if not state.table:
            return max(play_actions, key=card_value), None

        # Find current winner on table
        current_winner, _ = ListKit.winner(state.table, trump)

        # Find cards that beat the current winner
        winning_actions = [a for a in play_actions if CardKit.beats(ActionKit.value(a), trump, current_winner)]

        if winning_actions:
            return max(winning_actions, key=card_value), None
        return min(play_actions, key=card_value), None

