import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from ..core.agent import Agent
from ..kits import ActionKit
from ..types import Action, State, Log
from .network import PPONetwork


class PpoAgent(Agent):
    def __init__(
        self,
        network: PPONetwork,
        rng: Optional[np.random.Generator] = None,
        deterministic: bool = False,
    ):
        super().__init__()
        self.network = network
        self.device = next(network.parameters()).device
        self.deterministic = deterministic
        
        # Create separate optimizers for each network + shared embedding
        self.optimizers = {
            0: torch.optim.Adam(network.get_network_parameters(0), lr=1e-3, weight_decay=1e-5),
            1: torch.optim.Adam(network.get_network_parameters(1), lr=1e-3, weight_decay=1e-5),
            2: torch.optim.Adam(network.get_network_parameters(2), lr=1e-3, weight_decay=1e-5),
            3: torch.optim.Adam(network.get_network_parameters(3), lr=1e-3, weight_decay=1e-5),
            'shared': torch.optim.Adam(network.get_shared_embedding_parameters(), lr=5e-4, weight_decay=1e-5),
        }
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

        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(state_batch)
        if was_training:
            self.network.train()

        card_logits = outputs['card_policy'][0]  # [32]
        probs = torch.softmax(card_logits, dim=0)

        dist = torch.distributions.Categorical(probs)
        if self.deterministic:
            action_idx_tensor = torch.argmax(probs)
        else:
            action_idx_tensor = dist.sample()
        action_idx = action_idx_tensor.item()

        chosen_action = action_map[action_idx]

        log_prob = dist.log_prob(action_idx_tensor)
        value = outputs['value'][0]

        log = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'state': {
                'probabilities': state_batch['probabilities'].cpu().numpy(),
                'tables':        state_batch['tables'].cpu().numpy(),
                'history':       state_batch['history'].cpu().numpy(),
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
        entropy_coef   = 0.01
        max_grad_norm  = 0.5
        ppo_epochs     = 4
        batch_size     = 64

        # Record is a tuple: (player, state, action, reward, log) → log is r[4]
        probs = torch.tensor(
            np.array([r[4]['state']['probabilities'] for r in records]),
            dtype=torch.float32, device=self.device,
        ).squeeze(1)
        tables = torch.tensor(
            np.array([r[4]['state']['tables'] for r in records]),
            dtype=torch.long, device=self.device,
        ).squeeze(1)
        histories = torch.tensor(
            np.array([r[4]['state']['history'] for r in records]),
            dtype=torch.float32, device=self.device,
        ).squeeze(1)

        actions       = torch.tensor([r[4]['action_idx'] for r in records], dtype=torch.long,    device=self.device)
        old_log_probs = torch.tensor([r[4]['log_prob']   for r in records], dtype=torch.float32, device=self.device)
        masks         = torch.tensor(np.array([r[4]['mask'] for r in records]), dtype=torch.float32, device=self.device)

        if 'advantage' not in records[0][4] or 'return' not in records[0][4]:
            return {}

        advantages = torch.tensor([r[4]['advantage'] for r in records], dtype=torch.float32, device=self.device)
        returns    = torch.tensor([r[4]['return']    for r in records], dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Normalize returns to help value network learn (remember stats for denormalization if needed)
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        returns_normalized = (returns - returns_mean) / returns_std

        # Get table counts for each record to know which network was used
        batch_state_all = {
            'probabilities': probs,
            'tables':        tables,
            'history':       histories,
            'legal_actions': masks,
        }
        with torch.no_grad():
            outputs_all = self.network(batch_state_all)
            table_counts = outputs_all['table_counts']  # [N] - which network (0-3) for each sample

        # Diagnostic: Check value prediction vs target distribution
        with torch.no_grad():
            print(f"  Value predictions: mean={outputs_all['value'].mean():.2f}, std={outputs_all['value'].std():.2f}")
            print(f"  Value targets (original): mean={returns.mean():.2f}, std={returns.std():.2f}")
            print(f"  Value targets (normalized): mean={returns_normalized.mean():.2f}, std={returns_normalized.std():.2f}")

        # Adaptive value loss coefficient based on magnitude
        avg_value_loss_initial = nn.functional.mse_loss(outputs_all['value'].squeeze(-1), returns_normalized).item()
        if avg_value_loss_initial > 3.0:
            value_loss_coef = 2.0  # High loss - prioritize value learning
        elif avg_value_loss_initial > 1.0:
            value_loss_coef = 1.0  # Medium loss - balanced
        else:
            value_loss_coef = 0.5  # Low loss - fine-tuning
        
        print(f"  Using value_loss_coef={value_loss_coef:.1f} (initial value loss={avg_value_loss_initial:.2f})")

        # Initialize per-network metrics
        network_metrics = {i: {'policy_loss': [], 'value_loss': [], 'entropy': [], 'samples': 0} 
                          for i in range(4)}
        
        total_policy_loss = total_value_loss = total_entropy = total_loss = num_updates = 0
        indices = np.arange(len(records))

        for epoch in range(ppo_epochs):
            self.rng.shuffle(indices)

            for start in range(0, len(records), batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_state = {
                    'probabilities': probs[batch_idx],
                    'tables':        tables[batch_idx],
                    'history':       histories[batch_idx],
                    'legal_actions': masks[batch_idx],
                }

                outputs      = self.network(batch_state)
                card_logits  = outputs['card_policy']
                dist         = torch.distributions.Categorical(logits=card_logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy      = dist.entropy().mean()
                new_values   = outputs['value'].squeeze(-1)
                batch_table_counts = outputs['table_counts']

                ratio  = torch.exp(new_log_probs - old_log_probs[batch_idx])
                adv    = advantages[batch_idx]
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(new_values, returns_normalized[batch_idx])
                loss        = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

                # Zero gradients for all optimizers
                for opt in self.optimizers.values():
                    opt.zero_grad()
                
                loss.backward()
                
                # Clip gradients and step only the optimizers for networks used in this batch
                networks_used = set()
                for net_id in range(4):
                    mask = (batch_table_counts == net_id)
                    if mask.any():
                        networks_used.add(net_id)
                        # Track per-network metrics (using normalized returns)
                        net_policy_loss = -torch.min(surr1[mask], surr2[mask]).mean()
                        net_value_loss = nn.functional.mse_loss(new_values[mask], returns_normalized[batch_idx][mask])
                        network_metrics[net_id]['policy_loss'].append(net_policy_loss.item())
                        network_metrics[net_id]['value_loss'].append(net_value_loss.item())
                        network_metrics[net_id]['samples'] += mask.sum().item()
                
                # Clip gradients and step optimizers
                for net_id in networks_used:
                    nn.utils.clip_grad_norm_(self.network.get_network_parameters(net_id), max_grad_norm)
                    self.optimizers[net_id].step()
                
                # Always update shared embedding
                nn.utils.clip_grad_norm_(self.network.get_shared_embedding_parameters(), max_grad_norm)
                self.optimizers['shared'].step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                total_loss        += loss.item()
                num_updates       += 1

        # Compile metrics
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss':  total_value_loss  / num_updates,
            'entropy':     total_entropy     / num_updates,
            'total_loss':  total_loss        / num_updates,
        }
        
        # Add per-network metrics
        for net_id in range(4):
            if network_metrics[net_id]['policy_loss']:
                metrics[f'network_{net_id}_policy_loss'] = np.mean(network_metrics[net_id]['policy_loss'])
                metrics[f'network_{net_id}_value_loss'] = np.mean(network_metrics[net_id]['value_loss'])
                metrics[f'network_{net_id}_samples'] = network_metrics[net_id]['samples']
        
        return metrics

    # --- Helper Methods ---

    def _batch_state(self, state: dict) -> dict:
        """Convert a state dict to batched tensors ready for the network."""
        # probability is already a flat list of 128 floats (player * 32 + card)
        prob_tensor = torch.tensor(state['probability'], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 128]

        # Encode history (list of (player, action) tuples) → flat 128-float presence vector
        hist_vec = [0.0] * 128
        for player, action in state['history']:
            if ActionKit.is_play_card(action):
                card = ActionKit.value(action)
                hist_vec[player * 32 + card] = 1.0
        hist_tensor = torch.tensor(hist_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 128]

        # Pad table to length 4 with -1 (empty slots)
        table = list(state['table'])
        while len(table) < 4:
            table.append(-1)
        table_tensor = torch.tensor(table, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, 4]

        return {
            'probabilities': prob_tensor,
            'history':       hist_tensor,
            'tables':        table_tensor,
        }
