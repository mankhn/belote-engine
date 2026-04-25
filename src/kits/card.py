from ..ranks import Ranks
from ..suits import Suits
from ..types import Card, Trump, Suit, Rank
from .trump import TrumpKit

# Card values for different trump types. Indexed by rank (0-7).
_NON_TRUMP_VALUES  = {7: 11, 3: 10, 6: 4, 5: 3, 4: 2, 2: 0, 1: 0, 0: 0}
_NO_TRUMP_VALUES   = {7: 19, 3: 10, 6: 4, 5: 3, 4: 2, 2: 0, 1: 0, 0: 0}
_ALL_TRUMP_VALUES  = {4: 14, 2: 9, 7: 7, 3: 5, 6: 3, 5: 2, 1: 0, 0: 0}
_CARD_TRUMP_VALUES = {4: 20, 2: 14, 7: 11, 3: 10, 6: 4, 5: 3, 1: 0, 0: 0}


class CardKit:
    # A card is an int: suit * 8 + rank, range 0–31

    @staticmethod
    def make(suit: Suit, rank: Rank) -> Card:
        return suit * 8 + rank

    @staticmethod
    def suit(card: Card) -> Suit:
        return card // 8

    @staticmethod
    def rank(card: Card) -> Rank:
        return card % 8

    @staticmethod
    def str(card: Card) -> str:
        return f"{Ranks[card % 8]}{Suits[card // 8]}"

    @staticmethod
    def value(card: Card, trump: Trump) -> int:
        suit, rank = card // 8, card % 8
        if TrumpKit.is_no_trump(trump):
            return _NO_TRUMP_VALUES[rank]
        if TrumpKit.is_all_trump(trump):
            return _ALL_TRUMP_VALUES[rank]
        if TrumpKit.is_card_trump(trump):
            return _CARD_TRUMP_VALUES[rank] if suit == TrumpKit.suit(trump) else _NON_TRUMP_VALUES[rank]
        raise ValueError(f"Unknown trump: {trump}")

    @staticmethod
    def is_trump(card: Card, trump: Trump) -> bool:
        return TrumpKit.is_all_trump(trump) or \
               (TrumpKit.is_card_trump(trump) and card // 8 == TrumpKit.suit(trump))

    @staticmethod
    def beats(card: Card, trump: Trump, other: Card) -> bool:
        suit       = card // 8
        other_suit = other // 8
        if suit != other_suit:
            return TrumpKit.is_card_trump(trump) and suit == TrumpKit.suit(trump)
        diff = CardKit.value(card, trump) - CardKit.value(other, trump)
        return diff > 0 if diff != 0 else card % 8 > other % 8
