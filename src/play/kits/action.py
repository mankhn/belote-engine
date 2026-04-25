from ..types import Action

# Actions are tagged tuples: (type, *data)
# Types:
PASS             = 0
PLAY_CARD        = 1
SHOW_SET         = 2
ANNOUNCE_BELOTE  = 3


class ActionKit:

    @staticmethod
    def value(action: Action):
        return action[1]
    
    # Pass: (PASS,)
    
    @staticmethod
    def pass_() -> Action:
        return (PASS,)
    
    @staticmethod
    def is_pass(action: Action) -> bool:
        return action[0] == PASS

    # Play a card: (PLAY_CARD, card)

    @staticmethod
    def play_card(card: int) -> Action:
        return (PLAY_CARD, card)
    
    @staticmethod
    def is_play_card(action: Action) -> bool:
        return action[0] == PLAY_CARD
    
    # Show a set: (SHOW_SET, set)

    @staticmethod
    def show_set(set) -> Action:
        return (SHOW_SET, set)
    
    @staticmethod
    def is_show_set(action: Action) -> bool:
        return action[0] == SHOW_SET

    # Announce belote: (ANNOUNCE_BELOTE,)

    @staticmethod
    def announce_belote() -> Action:
        return (ANNOUNCE_BELOTE,)
    
    @staticmethod
    def is_announce_belote(action: Action) -> bool:
        return action[0] == ANNOUNCE_BELOTE

   





