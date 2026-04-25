from ..types import Objective

# Objectives are tagged tuples: (type, *data)
MAX_SCORE   = 0  # win by as much as possible
WIN_TRICK   = 1  # any positive margin
REACH_SCORE = 2  # hit a specific target score (e.g. 1000 points)


class ObjectiveKit:

    @staticmethod
    def value(objective: Objective) -> int:
        return objective[1]

    # Max score: (MAX_SCORE,)

    @staticmethod
    def max_score() -> Objective:
        return (MAX_SCORE,)
    
    @staticmethod
    def is_max_score(objective: Objective) -> bool:
        return objective[0] == MAX_SCORE

    # Win trick: (WIN_TRICK,)

    @staticmethod
    def win_trick() -> Objective:
        return (WIN_TRICK,)
    
    @staticmethod
    def is_win_trick(objective: Objective) -> bool:
        return objective[0] == WIN_TRICK

    # Reach score: (REACH_SCORE, points)

    @staticmethod
    def reach_score(points: int) -> Objective:
        return (REACH_SCORE, points)

    @staticmethod
    def is_reach_score(objective: Objective) -> bool:
        return objective[0] == REACH_SCORE





