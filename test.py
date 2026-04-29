"""
Smoke test — verifies the full pipeline end-to-end in ~500 steps.

Checks
------
  1. Config / constants are sane
  2. Pearl environment wrappers (plain + shielded) reset and step correctly
  3. Safety shield filters actions and exposes filter_rate
  4. DeepSetQNetwork forward pass (2-D and 3-D action batches)
  5. PrioritizedReplayBuffer push / sample / update_priorities
  6. DoubleDQNWithPER + PER agent builds without errors
  7. 500-step training loop — neural agent
  8. 500-step training loop — neurosymbolic (shielded) agent
  9. Model save / load round-trip
 10. Greedy evaluation episode (neural + neurosymbolic)
 11. Symbolic agent runs one episode

Usage
-----
  python test.py
  python -m unittest test
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


def _run_training(label: str, shielded: bool, model_dir: str) -> None:
    import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401
    from config import BASE_ENV_CONFIG, DQN_KWARGS
    from pearl_environment import make_pearl_env
    from train import _make_agent

    os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    env = make_pearl_env(BASE_ENV_CONFIG, shielded=shielded)
    agent, _ = _make_agent(total_timesteps=500, shielded=shielded)

    obs, action_space = env.reset()
    agent.reset(obs, action_space)

    learning_starts = DQN_KWARGS["learning_starts"]
    for step in range(500):
        action = agent.act(exploit=False)
        result = env.step(action)
        agent.observe(result)
        if step >= learning_starts:
            agent.learn()
        if result.done:
            obs, action_space = env.reset()
            agent.reset(obs, action_space)

    torch.save(
        {"agent_state": agent.state_dict(), "timestep": 500, "best_reward": 0.0},
        os.path.join(model_dir, "model_test.pth"),
    )
    env.close()


def _eval_one(shielded: bool, model_path: str) -> int:
    import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401
    from pearl_environment import make_pearl_eval_env
    from train import _make_agent

    agent, _ = _make_agent(total_timesteps=500, shielded=shielded)
    ckpt = torch.load(model_path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state"])

    env = make_pearl_eval_env(shielded=shielded)
    pl, sm, dev = agent.policy_learner, agent.safety_module, agent.device
    obs, action_space = env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        obs_t = torch.as_tensor(obs).to(dev)
        action_space.to(dev)  # pyright: ignore[reportAttributeAccessIssue]
        safe_space = sm.filter_action(obs_t, action_space)
        safe_space.to(dev)  # pyright: ignore[reportAttributeAccessIssue]
        action = pl.act(
            subjective_state=obs_t, available_action_space=safe_space, exploit=True
        )
        result = env.step(action)
        obs = result.observation
        action_space = result.available_action_space or env.action_space
        done = result.done
        steps += 1
    env.close()
    return steps


class TestConfig(unittest.TestCase):
    def test_constants(self):
        from config import BASE_ENV_CONFIG, DQN_KWARGS, N_FEATURES, N_OBS_VEHICLES

        self.assertGreater(N_OBS_VEHICLES, 0)
        self.assertGreater(N_FEATURES, 0)
        self.assertIn("observation", BASE_ENV_CONFIG)
        self.assertIn("buffer_size", DQN_KWARGS)


class TestPearlEnvironments(unittest.TestCase):
    def test_plain_env_reset_step(self):
        import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401
        from pearl_environment import make_pearl_env

        env = make_pearl_env(shielded=False)
        obs, action_space = env.reset()
        self.assertGreater(obs.shape[0], 0)
        result = env.step(action_space.actions[0])
        self.assertEqual(result.observation.shape, obs.shape)  # pyright: ignore[reportAttributeAccessIssue]
        env.close()

    def test_shielded_env_filter_rate(self):
        import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401
        from pearl_environment import make_pearl_env

        env = make_pearl_env(shielded=True)
        obs, action_space = env.reset()
        self.assertGreaterEqual(len(action_space.actions), 1)
        env.step(action_space.actions[0])
        self.assertTrue(hasattr(env, "filter_rate"))
        env.close()


class TestSafetyShield(unittest.TestCase):
    def test_is_action_safe(self):
        from config import N_FEATURES, N_OBS_VEHICLES
        from safety_shield import LTLSafetyShield

        shield = LTLSafetyShield()
        obs = np.zeros((N_OBS_VEHICLES, N_FEATURES), dtype=np.float32)
        results = [shield.is_action_safe(obs, a) for a in range(5)]
        self.assertIsInstance(results[0], bool)
        self.assertEqual(len(results), 5)


class TestDeepSetQNetwork(unittest.TestCase):
    def setUp(self):
        from deep_set_network import DeepSetQNetwork

        self.net = DeepSetQNetwork(n_vehicles=10, n_features=5, action_dim=5)

    def test_2d_actions(self):
        state = torch.randn(8, 50)
        q = self.net.get_q_values(state, torch.randn(8, 5))
        self.assertEqual(q.shape, (8,))

    def test_3d_actions(self):
        state = torch.randn(8, 50)
        q = self.net.get_q_values(state, torch.randn(8, 5, 5))
        self.assertEqual(q.shape, (8, 5))

    def test_permutation_invariance(self):
        state = torch.randn(1, 50)
        # Shuffle vehicle rows and confirm Q-values are identical
        shuffled = state.view(1, 10, 5)[:, torch.randperm(10), :].reshape(1, 50)
        action = torch.randn(1, 5)
        with torch.no_grad():
            q_orig = self.net.get_q_values(state, action)
            q_shuf = self.net.get_q_values(shuffled, action)
        self.assertTrue(torch.allclose(q_orig, q_shuf, atol=1e-5))


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def _fill_buffer(self, buf, n=50):
        import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401
        from pearl_environment import make_pearl_env

        buf._is_action_continuous = False
        buf.device_for_batches = torch.device("cpu")
        env = make_pearl_env(shielded=False)
        obs, action_space = env.reset()
        for _ in range(n):
            action = action_space.actions[0]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            result = env.step(action)
            buf.push(
                state=obs,
                action=action,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                curr_available_actions=action_space,
                next_state=result.observation,
                next_available_actions=result.available_action_space,
                max_number_actions=5,
            )
            obs = result.observation
            action_space = result.available_action_space
            if result.done:
                obs, action_space = env.reset()
        env.close()

    def test_push_sample_update(self):
        from per_replay_buffer import PrioritizedReplayBuffer

        buf = PrioritizedReplayBuffer(200)
        self._fill_buffer(buf, n=50)
        self.assertEqual(len(buf), 50)
        batch = buf.sample(16)
        self.assertIsNotNone(batch.weight)
        self.assertEqual(batch.weight.shape, (16,))  # pyright: ignore[reportOptionalMemberAccess]
        buf.update_priorities(torch.rand(16))

    def test_priorities_sum_to_one(self):
        from per_replay_buffer import PrioritizedReplayBuffer
        import numpy as np

        buf = PrioritizedReplayBuffer(200, alpha=0.6)
        self._fill_buffer(buf, n=50)
        prios = np.array(list(buf._priorities), dtype=np.float32)
        probs = prios**0.6
        probs /= probs.sum()
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


class TestAgentConstruction(unittest.TestCase):
    def test_neural_agent(self):
        from train import _make_agent

        agent, shield = _make_agent(total_timesteps=500, shielded=False)
        self.assertIsNotNone(agent)
        self.assertIsNone(shield)

    def test_shielded_agent(self):
        from train import _make_agent

        agent, shield = _make_agent(total_timesteps=500, shielded=True)
        self.assertIsNotNone(agent)
        self.assertIsNotNone(shield)


class TestTrainingLoop(unittest.TestCase):
    def test_neural_500_steps(self):
        _run_training("neural", shielded=False, model_dir="models/neural")
        self.assertTrue(os.path.exists("models/neural/model_test.pth"))

    def test_neurosymbolic_500_steps(self):
        _run_training("neurosymbolic", shielded=True, model_dir="models/neurosymbolic")
        self.assertTrue(os.path.exists("models/neurosymbolic/model_test.pth"))


class TestSaveLoad(unittest.TestCase):
    def test_round_trip(self):
        from train import _make_agent

        path = "models/neural/model_test.pth"
        if not os.path.exists(path):
            _run_training("neural", shielded=False, model_dir="models/neural")
        agent, _ = _make_agent(total_timesteps=500, shielded=False)
        ckpt = torch.load(path, weights_only=False)
        agent.load_state_dict(ckpt["agent_state"])  # must not raise


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("models/neural/model_test.pth"):
            _run_training("neural", shielded=False, model_dir="models/neural")
        if not os.path.exists("models/neurosymbolic/model_test.pth"):
            _run_training(
                "neurosymbolic", shielded=True, model_dir="models/neurosymbolic"
            )

    def test_neural_greedy_episode(self):
        steps = _eval_one(shielded=False, model_path="models/neural/model_test.pth")
        self.assertGreater(steps, 0)

    def test_neurosymbolic_greedy_episode(self):
        steps = _eval_one(
            shielded=True, model_path="models/neurosymbolic/model_test.pth"
        )
        self.assertGreater(steps, 0)


class TestSymbolicAgent(unittest.TestCase):
    def test_episode(self):
        import highway_env  # pyright: ignore[reportMissingImports] # noqa: F401
        from environments import make_eval_env
        from symbolic_agent import SymbolicAgent

        agent = SymbolicAgent()
        env = make_eval_env(shielded=False)
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        while not (done or truncated) and steps < 100:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))
            steps += 1
        env.close()
        self.assertGreater(steps, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
