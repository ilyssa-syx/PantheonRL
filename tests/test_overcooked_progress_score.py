"""Tests for PantheonRL Overcooked progress-score shaping."""

import unittest

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState

from overcookedgym.overcooked import (
    COUNTER_STAGING_BONUS,
    COUNTER_STAGING_CAP,
    DISH_TO_READY_POT_DISTANCE_COEF,
    HELD_SOUP_STALE_CAP,
    HELD_SOUP_DISTANCE_COEF,
    READY_SOUP_STALE_COEF,
    OvercookedMultiEnv,
)


class OvercookedProgressScoreTest(unittest.TestCase):
    def make_env(
        self,
        custom_dense_reward=False,
        custom_shaping_version=1,
        layout="simple",
        **kwargs
    ):
        return OvercookedMultiEnv(
            layout,
            custom_dense_reward=custom_dense_reward,
            custom_shaping_gamma=0.99,
            custom_shaping_scale=1.2,
            custom_shaping_version=custom_shaping_version,
            **kwargs
        )

    def state_with(self, env, held=None, counter=None, pot=None, counter_pos=None):
        state = env.mdp.get_standard_start_state()
        if held is not None:
            held_state = (
                ("onion", env.mdp.num_items_for_soup, env.mdp.soup_cooking_time)
                if held == "soup"
                else None
            )
            state.players[0].set_object(
                ObjectState(held, state.players[0].position, held_state)
            )
        if counter is not None:
            counter_pos = counter_pos or env.mdp.get_counter_locations()[0]
            counter_state = (
                ("onion", env.mdp.num_items_for_soup, env.mdp.soup_cooking_time)
                if counter == "soup"
                else None
            )
            state.add_object(ObjectState(counter, counter_pos, counter_state))
        if pot is not None:
            num_items, cook_time = pot
            pot_pos = env.mdp.get_pot_locations()[0]
            state.add_object(
                ObjectState("soup", pot_pos, ("onion", num_items, cook_time))
            )
        return state

    def test_progress_score_phases_match_harl(self):
        env = self.make_env()
        cooking_time = env.mdp.soup_cooking_time
        cases = [
            (self.state_with(env), 0),
            (self.state_with(env, held="onion"), 1),
            (self.state_with(env, pot=(1, 0)), 2),
            (self.state_with(env, pot=(2, 0)), 3),
            (self.state_with(env, pot=(3, 1)), 4),
            (self.state_with(env, held="dish", pot=(3, 1)), 5),
            (self.state_with(env, pot=(3, cooking_time)), 5),
            (self.state_with(env, held="dish", pot=(3, cooking_time)), 6),
            (self.state_with(env, held="soup"), 7),
        ]
        for state, expected_score in cases:
            with self.subTest(expected_score=expected_score):
                self.assertEqual(env._get_progress_score(state), expected_score)

    def test_throughput_potential_rewards_late_stage_progress(self):
        env = self.make_env(custom_dense_reward=True)
        cooking_time = env.mdp.soup_cooking_time
        held_soup_state = self.state_with(env, held="soup")
        held_soup_distance = env._nearest_distance(
            held_soup_state.players[0].position,
            env.mdp.get_serving_locations(),
        )
        held_dish_state = self.state_with(
            env, held="dish", pot=(3, cooking_time)
        )
        dish_to_ready_pot_distance = env._nearest_distance(
            held_dish_state.players[0].position,
            env._ready_soup_positions(held_dish_state),
        )

        self.assertAlmostEqual(env._potential(self.state_with(env, pot=(1, 0))), 0.3)
        self.assertAlmostEqual(env._potential(self.state_with(env, pot=(2, 0))), 0.6)
        self.assertAlmostEqual(env._potential(self.state_with(env, pot=(3, 1))), 1.0)
        self.assertAlmostEqual(
            env._potential(self.state_with(env, pot=(3, cooking_time))),
            1.2,
        )
        self.assertAlmostEqual(
            env._potential(held_dish_state),
            1.5
            - DISH_TO_READY_POT_DISTANCE_COEF * dish_to_ready_pot_distance,
        )
        self.assertAlmostEqual(
            env._potential(held_soup_state),
            2.0 - 0.05 * held_soup_distance,
        )

    def test_custom_shaping_uses_potential_based_reward(self):
        env = self.make_env(custom_dense_reward=True)
        prev_state = self.state_with(env, held="onion")
        next_state = self.state_with(env, pot=(1, 0))
        prev_phi = 0.0
        next_phi = 0.3
        self.assertAlmostEqual(
            env._calculate_custom_shaping(prev_state, next_state, False),
            1.2 * (0.99 * next_phi - prev_phi),
        )
        self.assertAlmostEqual(
            env._calculate_custom_shaping(prev_state, next_state, True),
            1.2 * (0.99 * next_phi - prev_phi),
        )
        self.assertAlmostEqual(
            env._calculate_custom_shaping(next_state, prev_state, False),
            1.2 * (0.99 * prev_phi - next_phi),
        )

    def test_custom_shaping_is_disabled_by_default(self):
        env = self.make_env()
        prev_state = self.state_with(env, held="onion")
        next_state = self.state_with(env, pot=(1, 0))
        self.assertEqual(
            env._calculate_custom_shaping(prev_state, next_state, False),
            0.0,
        )

    def test_stale_penalty_is_v2_extra_only(self):
        env = self.make_env(custom_dense_reward=True, custom_shaping_version=2)
        cooking_time = env.mdp.soup_cooking_time
        pot_pos = env.mdp.get_pot_locations()[0]
        ready_state = self.state_with(env, held="dish", pot=(3, cooking_time))
        dish_to_ready_pot_distance = env._nearest_distance(
            ready_state.players[0].position,
            env._ready_soup_positions(ready_state),
        )

        env.ready_soup_ages = {pot_pos: 10}

        self.assertAlmostEqual(
            env._potential(ready_state),
            1.5
            - DISH_TO_READY_POT_DISTANCE_COEF * dish_to_ready_pot_distance,
        )
        self.assertAlmostEqual(
            env._potential_v2_extra(ready_state),
            -5 * READY_SOUP_STALE_COEF,
        )

        held_state = self.state_with(env, held="soup")
        env.held_soup_ages = {0: 100}
        self.assertAlmostEqual(
            env._potential_v2_extra(held_state),
            -HELD_SOUP_STALE_CAP,
        )

    def test_custom_shaping_can_switch_to_v2_extra_scale(self):
        env = self.make_env(
            custom_dense_reward=True,
            custom_shaping_version=1,
            custom_shaping_extra_scale_v2=3.0,
            custom_shaping_version_switch_step=10,
            layout="random0",
        )
        if not env.non_edge_counters:
            self.skipTest("random0 layout has no non-edge counters")
        counter_pos = next(iter(env.non_edge_counters))
        prev_state = self.state_with(env)
        next_state = self.state_with(env, counter="soup", counter_pos=counter_pos)
        prev_base_phi = env._potential(prev_state)
        next_base_phi = env._potential(next_state)
        prev_extra_phi = env._potential_v2_extra(prev_state)
        next_extra_phi = env._potential_v2_extra(next_state)

        self.assertEqual(env._active_custom_shaping_version(), 1)
        self.assertAlmostEqual(
            env._calculate_custom_shaping(prev_state, next_state, False),
            1.2 * (0.99 * next_base_phi - prev_base_phi),
        )
        env.custom_shaping_elapsed_steps = 10
        self.assertEqual(env._active_custom_shaping_version(), 2)
        self.assertAlmostEqual(
            env._calculate_custom_shaping(prev_state, next_state, False),
            1.2 * (0.99 * next_base_phi - prev_base_phi)
            + 3.0 * (0.99 * next_extra_phi - prev_extra_phi),
        )

    def test_v2_uses_manhattan_distance_for_held_soup(self):
        env = self.make_env(custom_dense_reward=True, custom_shaping_version=2)
        state = self.state_with(env, held="soup")
        distance = env._nearest_distance(
            state.players[0].position, env.mdp.get_serving_locations()
        )

        self.assertAlmostEqual(
            env._potential_v2(state),
            2.0 - HELD_SOUP_DISTANCE_COEF * distance,
        )

    def test_v2_does_not_add_extra_dish_to_ready_pot_distance_shaping(self):
        env = self.make_env(custom_dense_reward=True, custom_shaping_version=2)
        cooking_time = env.mdp.soup_cooking_time
        state = self.state_with(env, held="dish", pot=(3, cooking_time))

        self.assertAlmostEqual(
            env._potential_v2(state),
            env._potential(state),
        )

    def test_interaction_signature_ignores_natural_cook_time_changes(self):
        env = self.make_env(custom_dense_reward=True, custom_shaping_version=2)
        cooking_time = env.mdp.soup_cooking_time
        prev_state = self.state_with(env, pot=(3, cooking_time - 1))
        next_state = self.state_with(env, pot=(3, cooking_time))

        self.assertEqual(
            env._interaction_signature(prev_state),
            env._interaction_signature(next_state),
        )
        self.assertAlmostEqual(
            env._calculate_useless_interact_penalty(
                prev_state,
                next_state,
                (Action.INTERACT, Action.STAY),
                0.0,
            ),
            -0.05,
        )

    def test_v2_matches_v1_before_late_stage(self):
        env = self.make_env(custom_dense_reward=True, custom_shaping_version=2)
        state = self.state_with(env, held="dish", pot=(2, 0))

        self.assertFalse(env._is_late_stage_shaping_state(state))
        self.assertAlmostEqual(env._potential_v2(state), env._potential(state))
        self.assertAlmostEqual(
            env._calculate_useless_interact_penalty(
                state,
                state,
                (Action.INTERACT, Action.STAY),
                0.0,
            ),
            0.0,
        )

    def test_useless_interact_penalty_caps_per_agent(self):
        env = self.make_env(custom_dense_reward=True, custom_shaping_version=2)
        state = self.state_with(env, held="soup")
        penalties = [
            env._calculate_useless_interact_penalty(
                state,
                state,
                (Action.INTERACT, Action.STAY),
                0.0,
            )
            for _ in range(20)
        ]

        self.assertAlmostEqual(penalties[0], -0.05)
        self.assertAlmostEqual(sum(penalties), -0.5)
        self.assertAlmostEqual(penalties[-1], 0.0)

    def test_counter_staging_bonus_only_applies_to_non_edge_counters(self):
        env = self.make_env(
            custom_dense_reward=True,
            custom_shaping_version=2,
            layout="random0",
        )
        if not env.non_edge_counters:
            self.skipTest("random0 layout has no non-edge counters")
        counter_pos = next(iter(env.non_edge_counters))
        state = self.state_with(env, counter="soup", counter_pos=counter_pos)

        self.assertAlmostEqual(
            env._potential_v2(state) - env._potential(state),
            min(COUNTER_STAGING_BONUS, COUNTER_STAGING_CAP),
        )


if __name__ == "__main__":
    unittest.main()
