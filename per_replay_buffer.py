"""
Prioritized Experience Replay buffer for Pearl.

Samples transitions proportionally to their TD-error priority, so rare
but high-signal events (near-crashes, tight lane-changes) are replayed
more often than routine cruising steps.

Reference: Schaul et al., "Prioritized Experience Replay" (2016).
"""

from collections import deque

import numpy as np
import torch

from pearl.replay_buffers.basic_replay_buffer import BasicReplayBuffer


class PrioritizedReplayBuffer(BasicReplayBuffer):
    """
    BasicReplayBuffer extended with priority-weighted sampling.

    Parameters
    ----------
    capacity             : max transitions stored
    alpha                : priority exponent  (0 = uniform, 1 = full priority)
    beta                 : IS-weight exponent (0 = no correction, 1 = full)
    beta_annealing_steps : anneal beta → 1 over this many learn() calls
    epsilon              : small constant added to every priority to keep
                           all transitions reachable
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(capacity)
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = (1.0 - beta) / beta_annealing_steps
        self._epsilon = epsilon
        self._priorities: deque[float] = deque([], maxlen=capacity)
        self._last_sampled_indices: list[int] = []

    # ── storage ──────────────────────────────────────────────────────────────

    def _store_transition(self, *args, **kwargs) -> None:
        max_prio = max(self._priorities) if self._priorities else 1.0
        super()._store_transition(*args, **kwargs)
        self._priorities.append(max_prio)

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, batch_size: int):
        n = len(self.memory)
        prios = np.array(self._priorities, dtype=np.float32)
        probs = prios ** self._alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        self._last_sampled_indices = indices.tolist()

        # Importance-sampling weights to correct for the sampling bias.
        weights = (n * probs[indices]) ** (-self._beta)
        weights /= weights.max()  # normalise so max weight = 1
        self._beta = min(1.0, self._beta + self._beta_increment)

        memory_list = list(self.memory)
        samples = [memory_list[i] for i in indices]
        batch = self._create_transition_batch(samples, self._is_action_continuous)
        batch.weight = torch.tensor(
            weights, dtype=torch.float32
        ).to(self.device_for_batches)
        return batch

    # ── priority update ───────────────────────────────────────────────────────

    def update_priorities(self, td_errors: torch.Tensor) -> None:
        """Update priorities for the transitions returned by the last sample()."""
        prios = list(self._priorities)
        for idx, err in zip(self._last_sampled_indices, td_errors.cpu().tolist()):
            prios[idx] = abs(float(err)) + self._epsilon
        self._priorities = deque(prios, maxlen=self.capacity)
