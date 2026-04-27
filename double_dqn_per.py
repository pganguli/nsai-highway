"""
DoubleDQN with Prioritized Experience Replay (PER).

Extends Pearl's DoubleDQN to:
  1. Apply importance-sampling (IS) weights from PrioritizedReplayBuffer to
     the Bellman loss, correcting for the priority-sampling bias.
  2. After each gradient update, propagate fresh TD errors back to the
     replay buffer so priorities stay current.

Usage
-----
    policy_learner = DoubleDQNWithPER(...)
    replay_buffer  = PrioritizedReplayBuffer(capacity)
    agent = PearlAgent(policy_learner=policy_learner, replay_buffer=replay_buffer, ...)
"""

from typing import Any

import torch
from pearl.policy_learners.sequential_decision_making.double_dqn import DoubleDQN
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch

from per_replay_buffer import PrioritizedReplayBuffer


class DoubleDQNWithPER(DoubleDQN):
    """DoubleDQN with IS-weighted loss and priority feedback."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_td_errors: torch.Tensor | None = None

    # ── IS-weighted learning step ─────────────────────────────────────────────

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        batch_size = batch.reward.shape[0]

        # Forward pass: predicted Q(s, a)
        predictions = self.forward(batch)  # (B,); also ticks target-net update

        # Targets: r + γ · max_{a' ∈ safe(s')} Q_target(s', a_online_best)
        with torch.no_grad():
            targets = (
                self.get_next_state_values(batch, batch_size)
                * self._discount_factor
                * (1 - batch.terminated.float())
            ) + batch.reward  # (B,)

        # Per-sample TD errors stored for priority update
        self._last_td_errors = (predictions.detach() - targets).abs()

        # IS-weighted MSE (batch.weight is None when using BasicReplayBuffer)
        if batch.weight is not None:
            loss = (batch.weight * (predictions - targets).pow(2)).mean()
        else:
            loss = torch.nn.functional.mse_loss(predictions, targets)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {"loss": self._last_td_errors.mean().item()}

    # ── learning loop with priority feedback ─────────────────────────────────

    def learn(self, replay_buffer: ReplayBuffer) -> dict[str, Any]:
        if len(replay_buffer) == 0:
            return {}

        batch_size = (
            len(replay_buffer)
            if self._batch_size == -1 or len(replay_buffer) < self._batch_size
            else self._batch_size
        )

        report: dict[str, list] = {}
        for _ in range(self._training_rounds):
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            if not isinstance(batch, TransitionBatch):
                continue
            batch = self.preprocess_batch(batch)
            single_report = self.learn_batch(batch)

            if (
                isinstance(replay_buffer, PrioritizedReplayBuffer)
                and self._last_td_errors is not None
            ):
                replay_buffer.update_priorities(self._last_td_errors)

            for k, v in single_report.items():
                report.setdefault(k, []).append(v)

        return report
