from src.types import Card, Trump
from src.kits import CardKit, TrumpKit, ListKit


def transform_canonical(trump: Trump, hands: list[list[Card]]):
    # Create canonical mapping based on player 0's hand
    suit_map = ListKit.suit_canonical_map(hands[0])

    # Transform trump and hands using canonical mapping
    if TrumpKit.is_card_trump(trump):
        trump = suit_map[TrumpKit.suit(trump)]
    hands = [
        [CardKit.make(suit_map[CardKit.suit(c)], CardKit.rank(c)) for c in hand]
        for hand in hands
    ]

    return trump, hands
