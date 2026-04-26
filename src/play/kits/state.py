from src.types import Card, Player
from src.kits import HistoryKit, ProbabilityKit
from ..types import Action, State
from .action import ActionKit


# Player index of "self" in the relative-to-current-player frame
PLAYER_SELF = 0

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

    def __getitem__(self, key):
        return self._s[key]

    def __setitem__(self, key, val):
        self._s[key] = val

    def __contains__(self, key):
        return key in self._s

    def get(self, key, default=None):
        return self._s.get(key, default)

    def observe(self, player: Player, action: Action):
        _observe(self._s, player, action)

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


def _observe(state: State, player: Player, action: Action):
    HistoryKit.record(state['history'], player, action)

    if ActionKit.is_play_card(action):
        card = ActionKit.value(action)
        state['table'].append(card)

        ProbabilityKit.update(state['probability'], player, card, 1.0)

        if player == PLAYER_SELF:
            state['holds'].remove(card)

        if len(state['table']) == 4:
            state['round'] += 1
            state['table'] = []
