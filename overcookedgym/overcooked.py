import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from pantheonrl.common.multiagentenv import SimultaneousEnv

class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(
        self,
        layout_name,
        ego_agent_idx=0,
        baselines=False,
        custom_dense_reward=False,
        custom_shaping_gamma=0.99,
        custom_shaping_scale=0.4,
        progress_weight=None,
    ):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()
        self.custom_dense_reward = bool(custom_dense_reward)
        self.custom_shaping_gamma = float(custom_shaping_gamma)
        self.custom_shaping_scale = float(custom_shaping_scale)
        # Deprecated compatibility knob; custom_shaping_scale is now the
        # single multiplier applied to progress-score differences.
        self.progress_weight = progress_weight
        self.cumulative_custom_shaped_rewards = 0.0

        DEFAULT_ENV_PARAMS = {
            "horizon": 400
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        try:
            mlp = MediumLevelPlanner.from_pickle_or_compute(
                self.mdp, NO_COUNTERS_PARAMS, force_compute=False
            )
        except ValueError as error:
            if "unsupported pickle protocol" not in str(error):
                raise
            print("Recomputing planner due to:", error)
            mlp = MediumLevelPlanner.from_pickle_or_compute(
                self.mdp, NO_COUNTERS_PARAMS, force_compute=True
            )

        self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()

    def _get_progress_score(self, state):
        """Return the highest global task-progress phase present in state."""
        pot_states = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(state)
        held_object_names = {
            player.held_object.name
            for player in state.players
            if player.held_object is not None
        }

        loose_ingredient_exists = bool(
            held_object_names.intersection(("onion", "tomato"))
            or counter_objects["onion"]
            or counter_objects["tomato"]
        )
        dish_in_transit = bool(
            "dish" in held_object_names or counter_objects["dish"]
        )
        plated_soup_in_transit = bool(
            "soup" in held_object_names or counter_objects["soup"]
        )

        ready_pots = pot_states["onion"]["ready"] + pot_states["tomato"]["ready"]
        cooking_pots = (
            pot_states["onion"]["cooking"] + pot_states["tomato"]["cooking"]
        )
        one_item_pots = (
            pot_states["onion"]["1_items"] + pot_states["tomato"]["1_items"]
        )
        two_item_pots = (
            pot_states["onion"]["2_items"] + pot_states["tomato"]["2_items"]
        )

        if plated_soup_in_transit:
            return 7
        if ready_pots and dish_in_transit:
            return 6
        if ready_pots or (cooking_pots and dish_in_transit):
            return 5
        if cooking_pots:
            return 4
        if two_item_pots:
            return 3
        if one_item_pots:
            return 2
        if loose_ingredient_exists:
            return 1
        return 0

    def _potential(self, state):
        return self._get_progress_score(state)

    def _calculate_custom_shaping(self, prev_state, next_state, done):
        if not self.custom_dense_reward:
            return 0.0
        prev_phi = self._potential(prev_state)
        next_phi = self._potential(next_state)
        return self.custom_shaping_scale * (
            self.custom_shaping_gamma * next_phi - prev_phi
        )

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(-high, high, dtype=np.float64)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        prev_state = self.base_env.state.deepcopy()
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)

        built_in_shaped_reward = float(info["shaped_r"])
        custom_shaped_reward = self._calculate_custom_shaping(
            prev_state, next_state, done
        )
        self.cumulative_custom_shaped_rewards += custom_shaped_reward
        shaped_reward = built_in_shaped_reward + custom_shaped_reward
        reward = sparse_reward + shaped_reward

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        step_info = {
            "sparse_reward": float(sparse_reward),
            "built_in_shaped_reward": built_in_shaped_reward,
            "custom_shaped_reward": float(custom_shaped_reward),
            "shaped_reward": float(shaped_reward),
            "total_reward": float(reward),
        }
        return (ego_obs, alt_obs), (reward, reward), done, step_info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.cumulative_custom_shaped_rewards = 0.0
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        pass
