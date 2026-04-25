from ..types import History


class HistoryKit:

    @staticmethod
    def make() -> History:
        return []

    @staticmethod
    def copy(history: History) -> History:
        return history.copy()

    @staticmethod
    def record(history: History, player: int, action: any):
        history.append((player, action))

    @staticmethod
    def played(history: History, action: any) -> bool:
        return any(a == action for _, a in history)

    @staticmethod
    def who_played(history: History, action: any) -> int:
        """Returns the player who played the action, or -1 if not found."""
        return next((p for p, a in history if a == action), -1)

    @staticmethod
    def actions_of(history: History, player: int) -> list[any]:
        """Returns all actions played by a given player, in order."""
        return [action for p, action in history if p == player]

    @staticmethod
    def all_played(history: History) -> list[any]:
        """Returns all played actions in order."""
        return [action for _, action in history]
