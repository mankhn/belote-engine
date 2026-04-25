from ..types import Probability, Player, Card

class ProbabilityKit:
    """Tracks card probabilities: flat list of length 128 (4 players × 32 cards).
    Values: 1.0 (has), 0.0 (not), 0-1 (prob). Index: player * 32 + card."""

    @staticmethod
    def make() -> Probability:
        return [0.25] * (4 * 32)

    @staticmethod
    def copy(matrix: Probability) -> Probability:
        return matrix.copy()

    @staticmethod
    def update(matrix: Probability, player: Player, card: Card, val: float) -> bool:
        """Updates probability for a card, redistributing remaining probability to others."""
        if sum(matrix[p * 32 + card] for p in range(4)) < 1e-6: return False

        # If setting to -1.0 (played), set EVERYONE to 0.0
        if abs(val + 1.0) < 1e-6:
            for p in range(4): matrix[p * 32 + card] = 0.0
            return True

        # If setting to 1.0 (certainty), clear others
        if abs(val - 1.0) < 1e-6:
            for p in range(4): matrix[p * 32 + card] = 0.0
            matrix[player * 32 + card] = 1.0
            return True

        if abs(matrix[player * 32 + card] - val) < 1e-6: return True

        others = [p for p in range(4) if p != player]
        others_vals = [matrix[p * 32 + card] for p in others]
        others_sum = sum(abs(v) for v in others_vals)

        matrix[player * 32 + card] = val
        remaining = 1.0 - abs(val)

        if others_sum < 1e-6:
            for p in others: matrix[p * 32 + card] = remaining / 3.0
        else:
            for p, v in zip(others, others_vals):
                matrix[p * 32 + card] = v * (remaining / others_sum)

        return True

    @staticmethod
    def extract(matrix: Probability, player: Player, card: Card, pct: float) -> bool:
        """Extracts probability from other players and adds to target player."""
        if sum(matrix[p * 32 + card] for p in range(4)) < 1e-6: return False

        others_sum = sum(matrix[p * 32 + card] for p in range(4) if p != player)
        if others_sum < 1e-6: return False

        return ProbabilityKit.update(matrix, player, card, matrix[player * 32 + card] + others_sum * pct)

