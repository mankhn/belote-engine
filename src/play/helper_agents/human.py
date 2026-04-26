from typing import List, Tuple, Optional
from src.types import Card
from src.kits import CardKit
from ..types import Action, State, Log
from ..core.agent import Agent
from ..kits import ActionKit


class Human(Agent):
    """Human player agent that prompts for input"""

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        """Prompt human player to choose an action"""
        valid_play_actions = [a for a in actions if ActionKit.is_play_card(a)]

        if not valid_play_actions:
            return actions[0] if actions else None, None

        valid_cards: List[Card] = [ActionKit.value(a) for a in valid_play_actions]
        player_hand: List[Card] = state.holds

        self._display_options(player_hand, valid_cards)
        return self._get_user_choice(player_hand, valid_cards, valid_play_actions), None

    def _display_options(self, player_hand: List[Card], valid_cards: List[Card]):
        """Display player's hand with valid card indices"""
        card_indices = []
        for i, card in enumerate(player_hand):
            if card in valid_cards:
                card_indices.append(str(i + 1))
            else:
                card_indices.append('.')

        print(f"Hands:[{' '.join(CardKit.str(c) for c in player_hand)}]")
        print(f"Input:  {'   '.join(card_indices)}", end=" : ")

    def _get_user_choice(self, player_hand: List[Card], valid_cards: List[Card], valid_actions: List[Action]) -> Action:
        """Get and validate user's card choice"""
        card_map = {i + 1: card for i, card in enumerate(player_hand)}

        while True:
            try:
                choice_input = input()
                if not choice_input:
                    continue
                choice = int(choice_input)
                if choice in card_map and card_map[choice] in valid_cards:
                    selected_card = card_map[choice]
                    for action in valid_actions:
                        if ActionKit.value(action) == selected_card:
                            return action
                else:
                    print("Invalid choice. Enter a valid number: ", end="")
            except ValueError:
                print("Please enter a number: ", end="")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                raise

