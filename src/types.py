# Player is an int: 0-3
type Player = int

# Suit is an int: 0-3 = clubs, diamonds, hearts, spades
type Suit = int

# Rank is an int: 0-7 = 7, 8, 9, J, Q, K, A
type Rank = int

# A card is an int: suit * 8 + rank, range 0–31
type Card = int

# Trump is an int: 0-3 = card trump (suit), 4 = no trump, 5 = all trump
type Trump = int

# flat list of length 128 (4 players × 32 cards)
type Probability = list[float]

# HistoryKit is an ordered list of (player, action) tuples.
type History = list[tuple[Player, any]]
