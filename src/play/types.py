from src.types import Card, Trump, Probability, History, Player


# Action is a tuple: (type, *data)
type Action = tuple[int, any]

# Objective is a tuple: (int, *data)
type Objective = tuple[int, any]

# State is personal from first person view.
type State = tuple[
    int,            # round: completed rounds
    list[Card],     # holds: cards in hand
    list[Card],     # table: cards on the table this round
    Probability,    # probability: Probability matrix
    History,        # history: History of (player, card) tuples
]

# Reward 
type Reward = tuple[
    int,            # Instant reward for the action taken
    int,            # Accrued reward for the current objective
]

# Log is a dictionary that can be used for training.
type Log = dict[str, any]

# Record 
type Record = tuple[
    Player,         # player who took the action
    State,          # state before the action
    Action,         # action taken
    Reward,         # reward received
    Log,            # log for training
]

# Game
type Game = tuple[
    list[Card],     # holds: cards in hand
    Trump,          # trump suit
    Player,         # player to play first
]

type Scores = tuple[
    int,            # score for team 1
    int,            # score for team 2
]