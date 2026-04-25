import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from ..core.agent import Agent
from ..kits import ActionKit
from ..types import Action, State, Log
from .network import PPONetwork


class PpoAgent(Agent):
    def __init__(self, network: PPONetwork, rng: Optional[np.random.Generator] = None):
        super().__init__()
        self.network = network
        self.device = next(network.parameters()).device
        self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def choose_action(self, state: State, actions: List[Action]) -> Tuple[Action, Optional[Log]]:
        """
        Choose an action based on the current game state.
        """
        play_card_actions = [a for a in actions if ActionKit.is_play_card(a)]

        if not play_card_actions:
            return actions[0], None

        # Build mask and action map
        action_map = {ActionKit.value(a): a for a in play_card_actions}
        mask = torch.full((32,), -float('inf'), device=self.device)
        for idx in action_map:
            mask[idx] = 0

        # Prepare batched state dict for network
        state_batch = self._batch_state(state)
        state_batch['legal_actions'] = mask.unsqueeze(0)

        with torch.no_grad():
            outputs = self.network(state_batch)

        card_logits = outputs['card_policy'][0]  # [32]
        probs = torch.softmax(card_logits, dim=0)

        dist = torch.distributions.Categorical(probs)
        action_idx_tensor = dist.sample()
        action_idx = action_idx_tensor.item()

        chosen_action = action_map[action_idx]

        log_prob = dist.log_prob(action_idx_tensor)
        value = outputs['value'][0]

        log = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'state': {
                'probability': state_batch['probability'].cpu().numpy(),
                'table':       state_batch['table'].cpu().numpy(),
                'history':     state_batch['history'].cpu().numpy(),
            },
            'action_idx': action_idx,
            'mask': mask.cpu().numpy(),
        }

        return chosen_action, log

    def learn(self, records: List) -> dict:
        """
        Train the agent based on the collected records.
        Assumes record.log contains 'advantage' and 'return' keys.
        """
        if not records:
            return {}

        # Hyperparameters
        clip_param     = 0.2
        value_loss_coef = 0.5
        entropy_coef   = 0.01
        max_grad_norm  = 0.5
        ppo_epochs     = 4
        batch_size     = 64

        probs = torch.tensor(
            np.array([r.log['state']['probability'] for r in records]),
            dtype=torch.float32, device=self.device,
        ).squeeze(1)
        tables = torch.tensor(
            np.array([r.log['state']['table'] for r in records]),
            dtype=torch.long, device=self.device,
        ).squeeze(1)
        histories = torch.tensor(
            np.array([r.log['state']['history'] for r in records]),
            dtype=torch.float32, device=self.device,
        ).squeeze(1)

        actions      = torch.tensor([r.log['action_idx'] for r in records], dtype=torch.long,    device=self.device)
        old_log_probs = torch.tensor([r.log['log_prob']   for r in records], dtype=torch.float32, device=self.device)
        masks        = torch.tensor(np.array([r.log['mask'] for r in records]), dtype=torch.float32, device=self.device)

        if 'advantage' not in records[0].log or 'return' not in records[0].log:
            return {}

        advantages = torch.tensor([r.log['advantage'] for r in records], dtype=torch.float32, device=self.device)
        returns    = torch.tensor([r.log['return']    for r in records], dtype=torch.float32, device=self.device)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = total_value_loss = total_entropy = total_loss = num_updates = 0
        indices = np.arange(len(records))

        for _ in range(ppo_epochs):
            self.rng.shuffle(indices)

            for start in range(0, len(records), batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_state = {
                    'probability':   probs[batch_idx],
                    'table':         tables[batch_idx],
                    'history':       histories[batch_idx],
                    'legal_actions': masks[batch_idx],
                }

                outputs      = self.network(batch_state)
                card_logits  = outputs['card_policy']
                dist         = torch.distributions.Categorical(logits=card_logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy      = dist.entropy().mean()
                new_values   = outputs['value'].squeeze(-1)

                ratio  = torch.exp(new_log_probs - old_log_probs[batch_idx])
                adv    = advantages[batch_idx]
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(new_values, returns[batch_idx])
                loss        = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                total_loss        += loss.item()
                num_updates       += 1

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss':  total_value_loss  / num_updates,
            'entropy':     total_entropy     / num_updates,
            'total_loss':  total_loss        / num_updates,
        }

    # --- Helper Methods ---

    def _batch_state(self, state: dict) -> dict:
        """Convert a state dict to batched tensors ready for the network."""
        # probability is already a flat list of 128 floats (player * 32 + card)
        prob_tensor = torch.tensor(state['probability'], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 128]

        # Encode history (list of (player, card) tuples) → flat 128-float presence vector
        hist_vec = [0.0] * 128
        for player, card in state['history']:
            hist_vec[player * 32 + card] = 1.0
        hist_tensor = torch.tensor(hist_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 128]

        # Pad table to length 4 with -1 (empty slots)
        table = list(state['table'])
        while len(table) < 4:
            table.append(-1)
        table_tensor = torch.tensor(table, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, 4]

        return {
            'probability': prob_tensor,
            'history':     hist_tensor,
            'table':       table_tensor,
        }
