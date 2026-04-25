from ..suits import Suits
from ..types import Trump


class TrumpKit:
    NO_TRUMP  = 4
    ALL_TRUMP = 5

    @staticmethod
    def is_no_trump(trump: Trump) -> bool:
        return trump == TrumpKit.NO_TRUMP

    @staticmethod
    def is_all_trump(trump: Trump) -> bool:
        return trump == TrumpKit.ALL_TRUMP

    @staticmethod
    def is_card_trump(trump: Trump) -> bool:
        return 0 <= trump <= 3

    @staticmethod
    def suit(trump: Trump) -> int:
        """Returns the trump suit (0-3). Only valid when is_card_trump."""
        return trump

    @staticmethod
    def str(trump: Trump) -> str:
        if trump == TrumpKit.NO_TRUMP:  return "No-Trump"
        if trump == TrumpKit.ALL_TRUMP: return "All-Trump"
        return f"Trump {Suits[trump]}"