"""Neural-network policies for discrete Soft Actor-Critic."""

from typing import Any, Dict, List, Optional, Tuple, Type

import torch as th
from gym import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class DiscreteActor(BasePolicy):
    """Categorical actor that outputs one logit per discrete action."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.logits_net = nn.Sequential(
            *create_mlp(
                features_dim,
                action_space.n,
                net_arch,
                activation_fn,
            )
        )

    def logits(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.logits_net(features)

    def action_distribution(
        self, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        logits = self.logits(obs)
        return F.log_softmax(logits, dim=1), F.softmax(logits, dim=1)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        logits = self.logits(obs)
        if deterministic:
            return logits.argmax(dim=1)
        return th.distributions.Categorical(logits=logits).sample()

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self(observation, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DiscreteQNetwork(BasePolicy):
    """Q-network that estimates a value for every discrete action."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.q_net = nn.Sequential(
            *create_mlp(
                features_dim,
                action_space.n,
                net_arch,
                activation_fn,
            )
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.q_net(features)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = True
    ) -> th.Tensor:
        return self(observation).argmax(dim=1)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DiscreteSACPolicy(BasePolicy):
    """Policy container holding the actor, twin critics, and target critics."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {"eps": 1e-4}

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "normalize_images": normalize_images,
        }

        self.actor = None
        self.qf1 = None
        self.qf2 = None
        self.qf1_target = None
        self.qf2_target = None
        self.q_optimizer = None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.qf1 = self.make_q_net()
        self.qf2 = self.make_q_net()
        self.qf1_target = self.make_q_net()
        self.qf2_target = self.make_q_net()

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.qf1_target.set_training_mode(False)
        self.qf2_target.set_training_mode(False)

        learning_rate = lr_schedule(1)
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=learning_rate,
            **self.optimizer_kwargs,
        )
        self.q_optimizer = self.optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=learning_rate,
            **self.optimizer_kwargs,
        )

    def make_actor(self) -> DiscreteActor:
        actor_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return DiscreteActor(**actor_args).to(self.device)

    def make_q_net(self) -> DiscreteQNetwork:
        q_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return DiscreteQNetwork(**q_args).to(self.device)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.actor(obs, deterministic=deterministic)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.qf1.set_training_mode(mode)
        self.qf2.set_training_mode(mode)
        self.training = mode

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


MlpPolicy = DiscreteSACPolicy
