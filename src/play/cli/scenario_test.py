"""
Belote Agent Scenario Testing

Tests a Belote agent against specific, well-known game scenarios.
Each scenario directly specifies probability, history, table and hand
rather than requiring a full 4-player deck setup.
"""

import os
import sys
import torch
import argparse
from typing import List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from src.kits.card import CardKit
from src.kits.trump import TrumpKit
from src.types import Card, Trump
from src.play.core.rules import Rules
from src.play.kits.state import StateKit
from src.play.kits.action import ActionKit
from src.play.ppo.network import PPONetwork
from src.play.ppo.agent import PpoAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANK_MAP  = {' 7': 0, ' 8': 1, ' 9': 2, '10': 3, ' J': 4, ' Q': 5, ' K': 6, ' A': 7}
SUIT_MAP  = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
SUIT_NAME = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
TRUMP_MAP = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'No-Trump': 4, 'All-Trump': 5}


def c(card_str: str) -> Card:
    """Parse a card string like ' 7♥' or '10♠' into a card int."""
    suit_char = card_str[-1]
    rank_str  = card_str[:-1]
    return CardKit.make(SUIT_MAP[suit_char], RANK_MAP[rank_str])


def cn(card: Card) -> str:
    return CardKit.str(card)


def build_state(
    hand:  List[str],             # player 0's current hand
    trump: str,                   # '♠' '♥' '♦' '♣' 'No-Trump' 'All-Trump'
    plays: List[Tuple[int, str]], # [(player, card_str), ...] in chronological order
) -> StateKit:
    """
    Build a StateKit by replaying plays through StateKit.observe().
    Probability, history and table are derived automatically.
    Only include plays by players 1-3; player 0's past plays are implicit
    (their cards are simply absent from hand).
    """
    hand_cards = [c(s) for s in hand]
    state = StateKit(StateKit.make(hand_cards))
    state['trump'] = TRUMP_MAP[trump]
    for player, card_str in plays:
        state.observe(player, ActionKit.play_card(c(card_str)))
    return state


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

class ScenarioTest:
    def __init__(
        self,
        name:           str,
        description:    str,
        hand:           List[str],              # player 0's current hand
        trump:          str,                    # '♠' '♥' '♦' '♣' 'No-Trump' 'All-Trump'
        plays:          List[Tuple[int, str]],  # [(player, card_str), ...] in order
        expected_cards: List[str],              # acceptable plays
        avoid_cards:    Optional[List[str]] = None,
    ):
        self.name           = name
        self.description    = description
        self.hand           = hand
        self.trump          = trump
        self.plays          = plays
        self.expected_cards = [c(s) for s in expected_cards]
        self.avoid_cards    = [c(s) for s in (avoid_cards or [])]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def create_test_scenarios() -> List[ScenarioTest]:
    scenarios = []

    # Scenario 1: Don't waste Ace on low card — opponent leads 8♥, you have A♥ and 7♥; should play 7♥ to save A♥ for later
    scenarios.append(ScenarioTest(
        name="dont_waste_ace_on_low",
        description="Opponent leads 8♥ — don't waste A♥ when you have 7♥ to cover",
        hand=[' A♥', '10♥', ' 7♥', ' 9♠', ' K♣', ' Q♠', ' 8♦', ' 7♦'],
        trump='♠',
        plays=[(1, ' 8♥')],
        expected_cards=[' A♥', '10♥'],
        avoid_cards=[' 7♥'],
    ))

    # Scenario 2: Don't waste 10♦ when King/Queen can cover opponent's 9♦
    scenarios.append(ScenarioTest(
        name="dont_waste_ten",
        description="Opponent leads 9♦ — don't waste 10♦ when K♦ or Q♦ can cover",
        hand=['10♦', ' K♦', ' Q♦', ' J♠', ' 8♣', ' 7♥', ' 8♥', ' 9♥'],
        trump='♠',
        plays=[(1, ' 9♦')],
        expected_cards=[' K♦', ' Q♦'],
        avoid_cards=['10♦'],
    ))

    # Scenario 3: Lead with trump Jack when you have it — it's the highest trump and can win the trick immediately
    scenarios.append(ScenarioTest(
        name="lead_with_trump_jack",
        description="Lead with trump Jack — it's the highest trump and can win the trick immediately",
        hand=[' J♥', ' 9♥', ' A♠', '10♣', ' K♦', ' Q♦', ' 7♠', ' 8♠'],
        trump='♥',
        plays=[],
        expected_cards=[' J♥', ' 9♥', ' A♠'],
        avoid_cards=['10♣', ' K♦', ' Q♦', ' 7♠', ' 8♠'],
    ))

    # Scenario 4: Cut with trump when you have no lead suit
    scenarios.append(ScenarioTest(
        name="cut_with_trump",
        description="Opponent leads 9♦, you have no diamonds but have trump 9♠ — cut with 9♠ to try to win the trick",
        hand=[' 9♠', ' 7♠', ' 7♥', ' 8♦', ' 9♦', '10♦', ' J♦', ' Q♦'],
        trump='♠',
        plays=[(1, ' A♣')],
        expected_cards=[' 9♠'],
        avoid_cards=[' 7♠'],
    ))

    # Scenario 5: Overtrump when opponent cuts with trump — opponent (player 2) cuts with 9♠, you have J♠ which can overtrump and win the trick immediately
    scenarios.append(ScenarioTest(
        name="overtrump_opponent",
        description="Opponent cuts with 9♠, you have J♠ which can overtrump and win the trick immediately",
        hand=[' A♥', ' K♥', ' Q♥', '10♥', ' 9♥', ' 8♥', ' 7♥', ' J♠'],
        trump='♠',
        plays=[(1, ' A♣'), (2, ' 9♠')],
        expected_cards=[' A♥'],
        avoid_cards=[],
    ))

    # Scenario 6 (Round 4): Overtrump with trump when opponent leads trump — opponent (player 1) leads 9♥ (trump), you have J♥ which can overtrump and win the trick immediately; must overtrump due to rules
    # P0 starting hand: [A♠, 10♠, 7♦, K♣, Q♣, J♣, 9♣, 8♥]
    # P0 plays K♣, Q♣, J♣, 9♣ in rounds 0-3 → remaining [A♠, 10♠, 7♦, 8♥]
    # Current trick: P1 leads 9♥ (trump); P0 has 8♥ (trump) and must overtrump with J♥ (not in hand, but rules force overtrump if possible)
    # Rules force P0 to overtrump: must play 8♥ (trump) to try to win the trick, even though it's not guaranteed to win against 9♥; if P0 had J♥, it would be the better overtrump choice
    scenarios.append(ScenarioTest(
        name="overtrump_trump_lead_round4",
        description="Round 4 — opponent leads 9♥ (trump), must overtrump with 8♥ (trump) to try to win the trick",
        hand=[' 9♥',  '10♥', ' A♠', ' 7♦', ' K♣', ' Q♣', ' J♣', ' 9♣'],
        trump='♥',
        plays=[
            (0, ' K♣'), (1, ' 7♣'), (2, ' 8♣'), (3, '10♣'),  # round 0
            (0, ' Q♣'), (1, ' A♣'), (2, ' 7♠'), (3, ' 8♠'),  # round 1
            (0, ' J♣'), (1, ' K♠'), (2, ' Q♠'), (3, ' J♠'),  # round 2
            (0, ' 9♣'), (1, ' 9♠'), (2, ' K♦'), (3, ' Q♦'),  # round 3
            (1, ' 8♥'), (2, ' J♥'), (3, ' A♦'),               # round 4 in progress
        ],
        expected_cards=['10♥'],
        avoid_cards=[' 9♥'],
    ))

    # Scenario 7 (Round 5): Opponent winning, no trump or lead suit — dump lowest card
    # P0 starting hand: [A♠, 10♠, 7♥, K♣, Q♣, J♣, 9♣, 8♠]
    # P0 plays K♣, Q♣, J♣, 9♣, 8♠ in rounds 0-4 → remaining [A♠, 10♠, 7♥]
    # Trump=♣; P0 has no clubs left and no diamonds → can play anything
    # Opponent (P1) winning with A♦ — dump 7♥ (0 pts) not A♠ (11 pts) or 10♠ (10 pts)
    scenarios.append(ScenarioTest(
        name="dump_low_opponent_winning_round5",
        description="Round 5 — opponent wins with A♦, no follow suit or trump, dump 7♥",
        hand=[' A♠', '10♠', ' 7♥', ' K♣', ' Q♣', ' J♣', ' 9♣', ' 8♠'],
        trump='♣',
        plays=[
            (0, ' K♣'), (1, ' 7♣'), (2, ' 8♣'), (3, '10♣'),  # round 0
            (0, ' Q♣'), (1, ' 7♠'), (2, ' K♠'), (3, ' Q♠'),  # round 1
            (0, ' J♣'), (1, ' J♠'), (2, ' K♦'), (3, ' Q♦'),  # round 2
            (1, ' 7♦'), (2, ' 8♦'), (3, ' J♦'), (0, ' 9♣'),  # round 3
            (1, ' 9♠'), (2, '10♦'), (3, ' 9♦'), (0, ' 8♠'),  # round 4
            (1, ' A♦'), (2, ' Q♥'), (3, ' J♥'),               # round 5 in progress
        ],
        expected_cards=[' 7♥'],
        avoid_cards=[' A♠', '10♠'],
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(agent: PpoAgent, rules: Rules, scenario: ScenarioTest):
    """Run one scenario. Returns (passed, chosen_card, message)."""
    state = build_state(
        hand=scenario.hand,
        trump=scenario.trump,
        plays=scenario.plays,
    )
    valid_actions = rules.actions(state)

    chosen_action, _ = agent.choose_action(state, valid_actions)

    if not ActionKit.is_play_card(chosen_action):
        return False, None, "Agent didn't play a card"

    chosen_card = ActionKit.value(chosen_action)

    if scenario.expected_cards and chosen_card not in scenario.expected_cards:
        expected_names = [cn(c) for c in scenario.expected_cards]
        return False, chosen_card, f"Expected one of {expected_names}, got {cn(chosen_card)}"

    if chosen_card in scenario.avoid_cards:
        avoid_names = [cn(c) for c in scenario.avoid_cards]
        return False, chosen_card, f"Should NOT play {cn(chosen_card)} (avoid: {avoid_names})"

    return True, chosen_card, f"Correctly played {cn(chosen_card)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test Belote Agent on Specific Scenarios")
    parser.add_argument("--model", type=str, default="models/model.pt")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train a model first.")
        return

    print(f"Loading model from {model_path}...")
    network = PPONetwork()
    network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    network.eval()

    agent = PpoAgent(network, deterministic=True)
    rules = Rules()
    scenarios = create_test_scenarios()

    print(f"\n{'='*80}")
    print(f"Running {len(scenarios)} Scenario Tests")
    print(f"{'='*80}\n")

    passed_tests = failed_tests = 0

    for i, scenario in enumerate(scenarios, 1):
        print(f"Test {i}: {scenario.name}")
        print(f"  Description: {scenario.description}")

        if args.verbose:
            print(f"  Hand:  {scenario.hand}")
            print(f"  Trump: {scenario.trump}")
            print(f"  Plays: {scenario.plays}")

        passed, chosen_card, message = run_scenario(agent, rules, scenario)

        if passed:
            passed_tests += 1
            print(f"  Result: ✓ PASS - {message}")
        else:
            failed_tests += 1
            print(f"  Result: ✗ FAIL - {message}")
        print()

    print(f"{'='*80}")
    print(f"Passed: {passed_tests}/{len(scenarios)}  "
          f"({passed_tests / len(scenarios) * 100:.1f}%)")
    print(f"{'='*80}")

    sys.exit(0 if failed_tests == 0 else 1)


if __name__ == "__main__":
    main()
