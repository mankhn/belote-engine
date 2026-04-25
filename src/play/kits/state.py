from src.types import Card, Player
from src.kits import HistoryKit, ProbabilityKit
from ..types import Action, State

# State is personal from first person view.
# holds:       list[Card]  — cards in hand
# table:       list[Card]  — cards on the table this round
# round:       int        — completed rounds
# probability: list       — Probability matrix
# history:     list       — History of (player, card) tuples

class StateKit:

    def __init__(self, state: State):
        object.__setattr__(self, '_s', state)

    def __getattr__(self, key):
        return self._s[key]

    def __setattr__(self, key, val):
        self._s[key] = val

    @staticmethod
    def make(holds: list[Card]) -> State:
        return {
            'holds':       list(holds),
            'table':       [],
            'round':       0,
            'probability': ProbabilityKit.make(),
            'history':     HistoryKit.make(),
        }

    @staticmethod
    def copy(state: State) -> State:
        return {
            'holds':       list(state['holds']),
            'table':       list(state['table']),
            'round':       state['round'],
            'probability': ProbabilityKit.copy(state['probability']),
            'history':     HistoryKit.copy(state['history']),
        }

    @staticmethod
    def observe(state: State, player: Player, action: Action):
        HistoryKit.record(state['history'], player, action)
        
        if Action.is_play_card(action):
            card = Action.card(action)
            state['table'].append(card)
            
            ProbabilityKit.update(state['probability'], player, card, 1.0)

            if player == Player.SELF:
                state['holds'].remove(card)

            if len(state['table']) == 4:
                state['round'] += 1
                state['table'] = []
