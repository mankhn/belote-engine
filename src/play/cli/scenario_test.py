"""
Belote Agent Scenario Testing

This script tests a Belote agent against specific, well-known game scenarios
to measure how accurately it makes strategic decisions.

Each test case is human-readable and tests a specific game strategy.
"""

import os
import sys
import torch
import argparse
from typing import List, Tuple

# Add the project root to the python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from src.kits.card import CardKit
from src.kits.trump import TrumpKit
from src.kits.probability import ProbabilityKit
from src.kits.history import HistoryKit
from src.types import Card, Trump, Player
from src.play.core.rules import Rules
from src.play.kits.state import StateKit
from src.play.kits.action import ActionKit
from src.play.ppo.network import PPONetwork
from src.play.ppo.agent import PpoAgent


class ScenarioTest:
    """Represents a single test scenario"""
    
    def __init__(self, name: str, description: str,
                 deck: List[List[Card]], trump: Trump,
                 played_actions: List[Tuple[Player, any]],
                 expected_cards: List[Card], avoid_cards: List[Card] = None):
        """
        Args:
            name: Short name for the test
            description: Human-readable description of the strategy being tested
            deck: 4 hands (lists of cards), one for each player [player0, player1, player2, player3]
            trump: Trump suit
            played_actions: List of (player, action) tuples that have already been played
            expected_cards: Cards the agent (player 0) should play (any of these is acceptable)
            avoid_cards: Cards the agent should NOT play (optional)
        """
        self.name = name
        self.description = description
        self.deck = deck
        self.trump = trump
        self.played_actions = played_actions
        self.expected_cards = expected_cards
        self.avoid_cards = avoid_cards or []


def make_card(rank_str: str, suit_str: str) -> Card:
    """Create a card from rank and suit strings"""
    rank_map = {' 7': 0, ' 8': 1, ' 9': 2, '10': 3, ' J': 4, ' Q': 5, ' K': 6, ' A': 7}
    suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
    return CardKit.make(suit_map[suit_str], rank_map[rank_str])


def card_name(card: Card) -> str:
    """Get human-readable name for a card"""
    return CardKit.str(card)


def create_test_scenarios() -> List[ScenarioTest]:
    """Create all test scenarios"""
    scenarios = []
    
    # Scenario 1: Don't waste Ace on opponent's low card when you have lower cards
    scenarios.append(ScenarioTest(
        name="dont_waste_ace_on_low",
        description="When opponent plays a low card (8♥), don't waste Ace if you have lower cards",
        deck=[
            # Player 0 (agent being tested)
            [
                make_card(' A', '♥'),  # Ace of hearts
                make_card('10', '♥'),  # 10 of hearts
                make_card(' 7', '♥'),  # 7 of hearts
                make_card(' 9', '♠'),
                make_card(' K', '♣'),
                make_card(' Q', '♠'),
                make_card(' 8', '♦'),
                make_card(' 7', '♦'),
            ],
            # Player 1 (opponent)
            [
                make_card(' 8', '♥'),
                make_card(' K', '♥'),
                make_card(' J', '♦'),
                make_card('10', '♦'),
                make_card(' 9', '♦'),
                make_card(' A', '♠'),
                make_card(' 7', '♣'),
                make_card(' 8', '♣'),
            ],
            # Player 2 (partner)
            [
                make_card(' 9', '♥'),
                make_card(' Q', '♥'),
                make_card(' J', '♥'),
                make_card(' A', '♦'),
                make_card(' K', '♦'),
                make_card('10', '♠'),
                make_card(' Q', '♣'),
                make_card('10', '♣'),
            ],
            # Player 3 (opponent)
            [
                make_card(' J', '♠'),
                make_card(' 8', '♠'),
                make_card(' 7', '♠'),
                make_card(' K', '♠'),
                make_card(' Q', '♦'),
                make_card(' 9', '♣'),
                make_card(' A', '♣'),
                make_card(' J', '♣'),
            ],
        ],
        trump=0,  # Spades trump
        played_actions=[
            (1, ActionKit.play_card(make_card(' 8', '♥'))),  # Player 1 plays 8♥
        ],
        expected_cards=[
            make_card(' 7', '♥'),  # Should play 7 of hearts
        ],
        avoid_cards=[
            make_card(' A', '♥'),  # Should NOT play Ace
            make_card('10', '♥'),  # Should NOT play 10
        ]
    ))
    
    # Scenario 2: Don't waste 10 when you can play lower
    scenarios.append(ScenarioTest(
        name="dont_waste_ten",
        description="When opponent plays 9♦, don't waste 10 if you have King (which is lower value)",
        deck=[
            # Player 0
            [
                make_card('10', '♦'),
                make_card(' K', '♦'),
                make_card(' Q', '♦'),
                make_card(' J', '♠'),
                make_card(' 8', '♣'),
                make_card(' 7', '♥'),
                make_card(' 8', '♥'),
                make_card(' 9', '♥'),
            ],
            # Player 1
            [
                make_card(' 9', '♦'),
                make_card(' 7', '♦'),
                make_card(' A', '♠'),
                make_card(' K', '♠'),
                make_card('10', '♣'),
                make_card(' A', '♥'),
                make_card('10', '♥'),
                make_card(' Q', '♥'),
            ],
            # Player 2
            [
                make_card(' A', '♦'),
                make_card(' J', '♦'),
                make_card(' 9', '♠'),
                make_card(' 8', '♠'),
                make_card(' 7', '♠'),
                make_card(' Q', '♣'),
                make_card(' 9', '♣'),
                make_card(' J', '♥'),
            ],
            # Player 3
            [
                make_card(' 8', '♦'),
                make_card('10', '♠'),
                make_card(' Q', '♠'),
                make_card(' A', '♣'),
                make_card(' K', '♣'),
                make_card(' J', '♣'),
                make_card(' 7', '♣'),
                make_card(' K', '♥'),
            ],
        ],
        trump=0,  # Spades trump
        played_actions=[
            (1, ActionKit.play_card(make_card(' 9', '♦'))),
        ],
        expected_cards=[
            make_card(' K', '♦'),  # King is lower value than 10
            make_card(' Q', '♦'),  # Queen is also lower
        ],
        avoid_cards=[
            make_card('10', '♦'),  # Don't waste the 10
        ]
    ))
    
    # Scenario 3: Lead with trump Jack when starting
    scenarios.append(ScenarioTest(
        name="lead_with_trump_jack",
        description="When you start and have trump Jack (highest trump), lead with it",
        deck=[
            # Player 0
            [
                make_card(' J', '♥'),  # Jack of hearts (trump)
                make_card(' 9', '♥'),  # 9 of hearts (trump)
                make_card(' A', '♠'),
                make_card('10', '♣'),
                make_card(' K', '♦'),
                make_card(' Q', '♦'),
                make_card(' 7', '♠'),
                make_card(' 8', '♠'),
            ],
            # Player 1
            [
                make_card(' A', '♥'),
                make_card(' K', '♥'),
                make_card(' Q', '♥'),
                make_card(' K', '♠'),
                make_card(' 9', '♠'),
                make_card(' A', '♣'),
                make_card(' K', '♣'),
                make_card(' A', '♦'),
            ],
            # Player 2
            [
                make_card('10', '♥'),
                make_card(' 8', '♥'),
                make_card(' 7', '♥'),
                make_card(' J', '♠'),
                make_card(' Q', '♠'),
                make_card(' Q', '♣'),
                make_card(' J', '♣'),
                make_card(' J', '♦'),
            ],
            # Player 3
            [
                make_card('10', '♠'),
                make_card(' 9', '♣'),
                make_card(' 8', '♣'),
                make_card(' 7', '♣'),
                make_card('10', '♦'),
                make_card(' 9', '♦'),
                make_card(' 8', '♦'),
                make_card(' 7', '♦'),
            ],
        ],
        trump=1,  # Hearts trump
        played_actions=[],  # No actions yet, player 0 starts
        expected_cards=[
            make_card(' J', '♥'),  # Should lead with trump Jack
        ],
        avoid_cards=[]
    ))
    
    # Scenario 4: Cut with trump when can't follow suit
    scenarios.append(ScenarioTest(
        name="cut_with_trump",
        description="When you can't follow suit, cut with a trump card",
        deck=[
            # Player 0
            [
                make_card(' J', '♠'),  # Jack of spades (trump)
                make_card(' 9', '♠'),  # 9 of spades (trump)
                make_card(' 7', '♥'),
                make_card(' 8', '♦'),
                make_card(' 9', '♦'),
                make_card('10', '♦'),
                make_card(' J', '♦'),
                make_card(' Q', '♦'),
            ],
            # Player 1
            [
                make_card(' A', '♣'),
                make_card(' K', '♣'),
                make_card('10', '♣'),
                make_card(' 9', '♣'),
                make_card(' A', '♥'),
                make_card('10', '♥'),
                make_card(' K', '♥'),
                make_card(' Q', '♥'),
            ],
            # Player 2
            [
                make_card(' 8', '♣'),
                make_card(' 7', '♣'),
                make_card(' Q', '♣'),
                make_card(' J', '♣'),
                make_card(' J', '♥'),
                make_card(' 9', '♥'),
                make_card(' 8', '♥'),
                make_card(' A', '♦'),
            ],
            # Player 3
            [
                make_card(' A', '♠'),
                make_card(' K', '♠'),
                make_card('10', '♠'),
                make_card(' Q', '♠'),
                make_card(' 8', '♠'),
                make_card(' 7', '♠'),
                make_card(' K', '♦'),
                make_card(' 7', '♦'),
            ],
        ],
        trump=0,  # Spades trump
        played_actions=[
            (1, ActionKit.play_card(make_card(' A', '♣'))),
        ],
        expected_cards=[
            make_card(' J', '♠'),  # Should cut with trump
            make_card(' 9', '♠'),
        ],
        avoid_cards=[
            make_card(' 7', '♥'),  # Don't throw away non-trump
            make_card(' 8', '♦'),
        ]
    ))
    
    # Scenario 5: Overtrump when opponent cuts
    scenarios.append(ScenarioTest(
        name="overtrump_opponent",
        description="When opponent cuts with 9♠, overtrump with Jack if you have it",
        deck=[
            # Player 0
            [
                make_card(' J', '♠'),  # Jack of spades (trump) - highest
                make_card(' A', '♥'),
                make_card(' K', '♥'),
                make_card(' Q', '♥'),
                make_card('10', '♥'),
                make_card(' 9', '♥'),
                make_card(' 8', '♥'),
                make_card(' 7', '♥'),
            ],
            # Player 1
            [
                make_card(' A', '♣'),
                make_card(' K', '♣'),
                make_card('10', '♣'),
                make_card(' Q', '♣'),
                make_card(' J', '♣'),
                make_card(' 9', '♣'),
                make_card(' 8', '♣'),
                make_card(' 7', '♣'),
            ],
            # Player 2
            [
                make_card(' 9', '♠'),
                make_card(' A', '♠'),
                make_card(' K', '♠'),
                make_card('10', '♠'),
                make_card(' Q', '♠'),
                make_card(' 8', '♠'),
                make_card(' 7', '♠'),
                make_card(' J', '♥'),
            ],
            # Player 3
            [
                make_card(' A', '♦'),
                make_card(' K', '♦'),
                make_card('10', '♦'),
                make_card(' Q', '♦'),
                make_card(' J', '♦'),
                make_card(' 9', '♦'),
                make_card(' 8', '♦'),
                make_card(' 7', '♦'),
            ],
        ],
        trump=0,  # Spades trump
        played_actions=[
            (1, ActionKit.play_card(make_card(' A', '♣'))),
            (2, ActionKit.play_card(make_card(' 9', '♠'))),  # Partner cuts with 9
        ],
        expected_cards=[
            make_card(' J', '♠'),  # Must overtrump
        ],
        avoid_cards=[]
    ))
    
    return scenarios


def run_scenario_test(agent: PpoAgent, rules: Rules, scenario: ScenarioTest) -> Tuple[bool, Card, str]:
    """
    Run a single scenario test
    
    Returns:
        (passed, chosen_card, message)
    """
    # Create initial state for player 0 (the agent we're testing)
    player0_hand = scenario.deck[0]
    raw_state = StateKit.make(player0_hand)
    raw_state['trump'] = scenario.trump
    state = StateKit(raw_state)
    
    # Replay all the played actions - state will observe and update itself
    for player, action in scenario.played_actions:
        # Convert player perspective: in state, player 0 is always "self"
        # So we need to adjust player numbers relative to player 0
        relative_player = player  # Already relative to player 0
        state.observe(relative_player, action)
    
    # Get valid actions from rules
    valid_actions = rules.actions(state)
    
    # Agent chooses action
    chosen_action, _ = agent.choose_action(state, valid_actions)
    
    if not ActionKit.is_play_card(chosen_action):
        return False, None, "Agent didn't play a card"
    
    chosen_card = ActionKit.value(chosen_action)
    
    # Check if chosen card is expected
    if scenario.expected_cards and chosen_card not in scenario.expected_cards:
        expected_names = [card_name(c) for c in scenario.expected_cards]
        message = f"Expected one of {expected_names}, got {card_name(chosen_card)}"
        passed = False
    elif chosen_card in scenario.avoid_cards:
        avoid_names = [card_name(c) for c in scenario.avoid_cards]
        message = f"Should NOT play {card_name(chosen_card)} (avoid: {avoid_names})"
        passed = False
    else:
        message = f"Correctly played {card_name(chosen_card)}"
        passed = True
    
    return passed, chosen_card, message


def main():
    parser = argparse.ArgumentParser(description="Test Belote Agent on Specific Scenarios")
    parser.add_argument("--model", type=str, default="models/model.pt", 
                       help="Path to the trained model file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output for each test")
    args = parser.parse_args()
    
    # Load model
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train a model first.")
        return
    
    print(f"Loading model from {model_path}...")
    network = PPONetwork()
    network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    network.eval()
    
    agent = PpoAgent(network)
    rules = Rules()
    
    # Create and run scenarios
    scenarios = create_test_scenarios()
    
    print(f"\n{'='*80}")
    print(f"Running {len(scenarios)} Scenario Tests")
    print(f"{'='*80}\n")
    
    passed_tests = 0
    failed_tests = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Test {i}: {scenario.name}")
        print(f"  Description: {scenario.description}")
        
        if args.verbose:
            print(f"  Player 0 hand: {[card_name(c) for c in scenario.deck[0]]}")
            print(f"  Trump: {TrumpKit.str(scenario.trump)}")
            if scenario.played_actions:
                print(f"  Played actions:")
                for player, action in scenario.played_actions:
                    if ActionKit.is_play_card(action):
                        card = ActionKit.value(action)
                        print(f"    Player {player}: {card_name(card)}")
        
        passed, chosen_card, message = run_scenario_test(agent, rules, scenario)
        
        if passed:
            passed_tests += 1
            status = "✓ PASS"
        else:
            failed_tests += 1
            status = "✗ FAIL"
        
        print(f"  Result: {status} - {message}")
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Total Tests: {len(scenarios)}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Accuracy: {(passed_tests / len(scenarios) * 100):.1f}%")
    print(f"{'='*80}")
    
    # Return non-zero exit code if any tests failed
    sys.exit(0 if failed_tests == 0 else 1)


if __name__ == "__main__":
    main()
