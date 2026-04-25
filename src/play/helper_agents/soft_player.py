from typing import List, Tuple, Optional
from src.types import Action
from src.kits import ListKit
from ..types import Agent, State, Log
from ..kits import ActionKit


class SoftPlayer(Agent):
    """Soft player agent that tries to lose the trick"""
    
    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        """Choose the card that loses the trick, or lowest value card"""
        # Filter for play card actions only
        play_actions = [action for action in actions if ActionKit.is_play_card(action)]
        
        if not play_actions:
            # If no play card actions available, return first action
            return actions[0] if actions else None, None
        
        # If leading (table empty), play lowest value card
        if not state.table:
            best_action = min(play_actions, key=lambda a: a.card.value(state.trump))
            return best_action, None

        # Find current winner on table
        current_winner, _ = ListKit.winner(state.table, state.trump)
        
        # Find cards that DO NOT beat the current winner
        losing_actions = [a for a in play_actions if not a.card.beats(state.trump, current_winner)]
        
        if losing_actions:
            # If we can lose, play the lowest value losing card
            best_action = min(losing_actions, key=lambda a: a.card.value(state.trump))
        else:
            # If we must win, play the lowest value winning card (win cheaply)
            best_action = min(play_actions, key=lambda a: a.card.value(state.trump))
        
        return best_action, None
