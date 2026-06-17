"""Discrete Soft Actor-Critic implemented on Stable-Baselines3 interfaces."""

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from .policies import DiscreteSACPolicy, MlpPolicy


SelfDiscreteSAC = TypeVar("SelfDiscreteSAC", bound="DiscreteSAC")


class DiscreteSAC(OffPolicyAlgorithm):
    """Soft Actor-Critic for finite discrete action spaces."""

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DiscreteSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 20_000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        target_entropy_scale: float = 0.89,
        target_update_interval: int = 8_000,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        if target_entropy_scale <= 0:
            raise ValueError("target_entropy_scale must be positive")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")

        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.target_entropy_scale = target_entropy_scale
        self.target_update_interval = target_update_interval
        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        self.ent_coef_tensor = None
        self._n_calls = 0

        self.actor = None
        self.qf1 = None
        self.qf2 = None
        self.qf1_target = None
        self.qf2_target = None
        self.q_optimizer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        self.batch_norm_stats = (
            get_parameters_by_name(self.qf1, ["running_"])
            + get_parameters_by_name(self.qf2, ["running_"])
        )
        self.batch_norm_stats_target = (
            get_parameters_by_name(self.qf1_target, ["running_"])
            + get_parameters_by_name(self.qf2_target, ["running_"])
        )

        if self.target_entropy == "auto":
            self.target_entropy = (
                self.target_entropy_scale * np.log(self.action_space.n)
            )
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                if init_value <= 0:
                    raise ValueError(
                        "The initial entropy coefficient must be positive"
                    )
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef],
                lr=self.lr_schedule(1),
                eps=1e-4,
            )
        else:
            self.ent_coef_tensor = th.tensor(
                float(self.ent_coef), device=self.device
            )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.qf1 = self.policy.qf1
        self.qf2 = self.policy.qf2
        self.qf1_target = self.policy.qf1_target
        self.qf2_target = self.policy.qf2_target
        self.q_optimizer = self.policy.q_optimizer

    def _current_ent_coef(self) -> th.Tensor:
        if self.log_ent_coef is not None:
            return th.exp(self.log_ent_coef.detach())
        return self.ent_coef_tensor

    def get_ent_coef(self) -> float:
        """Return the current entropy coefficient as a Python float."""
        return float(self._current_ent_coef().item())

    def _on_step(self) -> None:
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(
                self.qf1.parameters(), self.qf1_target.parameters(), self.tau
            )
            polyak_update(
                self.qf2.parameters(), self.qf2_target.parameters(), self.tau
            )
            polyak_update(
                self.batch_norm_stats, self.batch_norm_stats_target, 1.0
            )
        self.logger.record("rollout/ent_coef", self.get_ent_coef())

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.q_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        actor_losses = []
        critic_losses = []
        ent_coef_losses = []
        ent_coefs = []
        entropies = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            ent_coef = self._current_ent_coef()
            ent_coefs.append(ent_coef.item())

            with th.no_grad():
                next_log_probs, next_probs = self.actor.action_distribution(
                    replay_data.next_observations
                )
                next_min_q = th.minimum(
                    self.qf1_target(replay_data.next_observations),
                    self.qf2_target(replay_data.next_observations),
                )
                next_soft_value = (
                    next_probs * (next_min_q - ent_coef * next_log_probs)
                ).sum(dim=1, keepdim=True)
                target_q = replay_data.rewards + (
                    1 - replay_data.dones
                ) * self.gamma * next_soft_value

            actions = replay_data.actions.long()
            current_qf1 = self.qf1(replay_data.observations).gather(
                dim=1, index=actions
            )
            current_qf2 = self.qf2(replay_data.observations).gather(
                dim=1, index=actions
            )
            critic_loss = 0.5 * (
                F.mse_loss(current_qf1, target_q)
                + F.mse_loss(current_qf2, target_q)
            )
            critic_losses.append(critic_loss.item())

            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self.q_optimizer.step()

            log_probs, probs = self.actor.action_distribution(
                replay_data.observations
            )
            with th.no_grad():
                min_q = th.minimum(
                    self.qf1(replay_data.observations),
                    self.qf2(replay_data.observations),
                )
            actor_loss = (
                probs * (ent_coef * log_probs - min_q)
            ).sum(dim=1).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            entropy = -(probs.detach() * log_probs.detach()).sum(
                dim=1, keepdim=True
            )
            entropies.append(entropy.mean().item())
            if self.ent_coef_optimizer is not None:
                alpha = th.exp(self.log_ent_coef)
                ent_coef_loss = (
                    alpha * (entropy - self.target_entropy)
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/entropy", np.mean(entropies))
        if ent_coef_losses:
            self.logger.record(
                "train/ent_coef_loss", np.mean(ent_coef_losses)
            )

    def learn(
        self: SelfDiscreteSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DiscreteSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDiscreteSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "actor",
            "qf1",
            "qf2",
            "qf1_target",
            "qf2_target",
            "q_optimizer",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = [
            "policy",
            "policy.actor.optimizer",
            "policy.q_optimizer",
        ]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            saved_pytorch_variables = ["log_ent_coef"]
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
