"""Core tests for the SB3-compatible Discrete SAC implementation."""

from pathlib import Path
import tempfile
import unittest

import gym
import numpy as np
import torch as th

from pantheonrl.algos.discrete_sac import DiscreteSAC


class TinyDiscreteEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        reward = float(action == 1)
        return np.zeros(4, dtype=np.float32), reward, True, {}


class DiscreteSACTest(unittest.TestCase):
    def make_model(self, env):
        return DiscreteSAC(
            "MlpPolicy",
            env,
            learning_starts=1,
            buffer_size=100,
            batch_size=4,
            train_freq=1,
            target_update_interval=2,
            seed=0,
            device="cpu",
        )

    def test_actor_and_critics_output_all_actions(self):
        env = TinyDiscreteEnv()
        model = self.make_model(env)
        obs = th.zeros((5, 4), dtype=th.float32)
        log_probs, probs = model.actor.action_distribution(obs)

        self.assertEqual(tuple(log_probs.shape), (5, 3))
        self.assertEqual(tuple(model.qf1(obs).shape), (5, 3))
        self.assertEqual(tuple(model.qf2(obs).shape), (5, 3))
        th.testing.assert_close(probs.sum(dim=1), th.ones(5))

    def test_training_updates_both_learners_and_alpha(self):
        env = TinyDiscreteEnv()
        model = self.make_model(env)
        actor_before = [
            parameter.detach().clone() for parameter in model.actor.parameters()
        ]
        q_before = [
            parameter.detach().clone() for parameter in model.qf1.parameters()
        ]
        alpha_before = model.get_ent_coef()

        model.learn(total_timesteps=20)

        self.assertEqual(model.replay_buffer.size(), 20)
        self.assertGreater(model._n_updates, 0)
        self.assertNotEqual(model.get_ent_coef(), alpha_before)
        self.assertTrue(
            any(
                not th.equal(before, after)
                for before, after in zip(actor_before, model.actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not th.equal(before, after)
                for before, after in zip(q_before, model.qf1.parameters())
            )
        )

    def test_save_load_preserves_deterministic_prediction(self):
        env = TinyDiscreteEnv()
        model = self.make_model(env)
        model.learn(total_timesteps=10)
        observation = env.reset()
        expected, _ = model.predict(observation, deterministic=True)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "model"
            model.save(path)
            loaded = DiscreteSAC.load(path, env=env, device="cpu")

        actual, _ = loaded.predict(observation, deterministic=True)
        self.assertEqual(int(actual), int(expected))
        self.assertAlmostEqual(loaded.get_ent_coef(), model.get_ent_coef())


if __name__ == "__main__":
    unittest.main()
