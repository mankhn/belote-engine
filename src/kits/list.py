from ..types import Card, Trump
from .card import CardKit


class ListKit:

    @staticmethod
    def winner(cards: list[Card], trump: Trump) -> tuple[int | None, int | None]:
        if not cards:
            return None, None

        best_idx, best_card = 0, cards[0]
        for i in range(1, len(cards)):
            if CardKit.beats(cards[i], trump, best_card):
                best_idx, best_card = i, cards[i]
        return best_card, best_idx

    @staticmethod
    def value(cards: list[Card], trump: Trump) -> int:
        return sum(CardKit.value(c, trump) for c in cards)

    @staticmethod
    def sort(cards: list[Card], trump: Trump) -> list[Card]:
        suit_stats = {}
        for c in cards:
            s = CardKit.suit(c)
            if s not in suit_stats:
                suit_stats[s] = {'max': 0, 'sum': 0, 'count': 0}
            v = CardKit.value(c, trump)
            suit_stats[s]['max'] = max(suit_stats[s]['max'], v)
            suit_stats[s]['sum'] += v
            suit_stats[s]['count'] += 1

        cards.sort(key=lambda c: (
            not CardKit.is_trump(c, trump),        # Trump cards first
            -suit_stats[CardKit.suit(c)]['max'],   # Sort suits by max value (highest first)
            -suit_stats[CardKit.suit(c)]['sum'],   # If equal, by sum (highest first)
            -suit_stats[CardKit.suit(c)]['count'], # If equal, by count (most cards first)
            CardKit.suit(c),                       # If still equal, by suit number (ascending)
            -CardKit.value(c, trump),              # Within suit, highest value first
            -CardKit.rank(c),                      # Tiebreaker
        ))
        return cards

    @staticmethod
    def suit_canonical_map(cards: list[Card]) -> dict[int, int]:
        strengths = [0.0] * 4
        counts    = [0]   * 4
        for c in cards:
            s = CardKit.suit(c)
            counts[s]    += 1
            strengths[s] += CardKit.rank(c)

        for s in range(4):
            if counts[s]:
                strengths[s] /= counts[s]

        ordered = sorted(range(4), key=lambda s: (-strengths[s], s))
        return {orig: canon for canon, orig in enumerate(ordered)}