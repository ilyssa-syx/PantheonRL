import gym
from stable_baselines3 import DQN
from pantheonrl.common.agents import OffPolicyAgent
import overcookedgym

env = gym.make("OvercookedMultiEnv-v0", layout_name="simple")
partner_env = env.getDummyEnv(1)

partner_model = DQN(
    "MlpPolicy",
    partner_env,
    verbose=1,
    seed=0,
)

partner_agent = OffPolicyAgent(partner_model)
env.add_partner_agent(partner_agent)

ego_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0,
)

ego_model.learn(total_timesteps=100_000)

ego_model.save("results/models/ego_dqn_seed0")
partner_model.save("results/models/partner_dqn_seed0")