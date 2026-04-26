import random
import sys
import os
import pickle
import argparse

# Add the project root to the python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from src.play.cli.utils.transformer import transform_canonical
from src.kits import ListKit, TrumpKit
from src.play.core.simulator import Simulator
from src.play.core.rules import Rules
from src.play.helper_agents.human import Human


def main():
    parser = argparse.ArgumentParser(description="Record Belote Game")
    parser.add_argument("--output", type=str, default="records/record-001.pkl", help="Path to save the record file (relative paths resolve to project root)")
    args = parser.parse_args()

    # Randomly choose trump (0-3 = card trump, 4 = no-trump, 5 = all-trump)
    trump = random.randint(0, 3)

    # Hands: shuffle 32 cards into 4 hands of 8, then sort each
    deck = list(range(32))
    random.shuffle(deck)
    hands = [ListKit.sort(deck[i * 8:(i + 1) * 8], trump) for i in range(4)]

    # All Human
    agents = [Human() for _ in range(4)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)

    # Transform to canonical form
    trump, hands = transform_canonical(trump, hands)

    print("")
    print("Starting Belote Game Recording...")
    print(f"{TrumpKit.str(trump)} | You will play for all 4 players.")

    # Run simulation
    start_player = random.randint(0, 3)
    scores, records = simulator.simulate((hands, trump, start_player))

    print("\nGame Over!")
    print(f"Total Scores: {scores}")

    # Convert StateKit instances inside records to plain dicts for pickling
    serializable_records = [
        (player, state._s, action, reward, log)
        for (player, state, action, reward, log) in records
    ]

    # Save the result (hands, trump, start_player, records)
    save_path = args.output
    if not os.path.isabs(save_path):
        save_path = os.path.join(PROJECT_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "hands":        hands,
            "trump":        trump,
            "start_player": start_player,
            "records":      serializable_records,
        }, f)
    print(f"Game recorded to {save_path}")


if __name__ == "__main__":
    main()

