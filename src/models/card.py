from src.models.trump import Trump, TrumpMode
from src.ranks import Ranks
from src.suits import Suits

class Card:
    def __init__(self, suit: int, rank: int):
        assert 0 <= suit <= 3 and 0 <= rank <= 7
        self.suit, self.rank = suit, rank
    
    def __iter__(self):
        return iter((self.suit, self.rank))
    
    def __repr__(self):
        return f"{Ranks[self.rank]}{Suits[self.suit]}"

    def __int__(self) -> int:
        """Convert card to index 0-31."""
        return self.suit * 8 + self.rank

    def is_trump(self, trump: Trump) -> bool:
        return (trump.mode == TrumpMode.AllTrump) or \
               (trump.mode == TrumpMode.Regular and self.suit == trump.suit)

    def beats(self, trump: Trump, next: "Card") -> bool:
        if self.suit != next.suit:
            return trump.mode == TrumpMode.Regular and self.suit == trump.suit
        
        diff = self.value(trump) - next.value(trump)
        return diff > 0 if diff != 0 else self.rank > next.rank
    
    _NO_TRUMP_VALUES = {7: 19, 3: 10, 6: 4, 5: 3, 4: 2, 2: 0, 1: 0, 0: 0}
    _ALL_TRUMP_VALUES = {4: 14, 2: 9, 7: 7, 3: 5, 6: 3, 5: 2, 1: 0, 0: 0}
    _REGULAR_TRUMP_VALUES = {4: 20, 2: 14, 7: 11, 3: 10, 6: 4, 5: 3, 1: 0, 0: 0}
    _REGULAR_NON_TRUMP_VALUES = {7: 11, 3: 10, 6: 4, 5: 3, 4: 2, 2: 0, 1: 0, 0: 0}

    def value(self, trump: Trump) -> int:
        # No Trump mode
        if trump.mode == TrumpMode.NoTrump:
            return self._NO_TRUMP_VALUES[self.rank]

        # All Trump mode
        elif trump.mode == TrumpMode.AllTrump:
            return self._ALL_TRUMP_VALUES[self.rank]

        # Regular (Trump suit) mode
        elif trump.mode == TrumpMode.Regular:
            if self.suit == trump.suit:
                return self._REGULAR_TRUMP_VALUES[self.rank]
            else:
                return self._REGULAR_NON_TRUMP_VALUES[self.rank]

        # Default fallback
        raise ValueError(f"Unknown trump mode: {trump.mode}")
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))


