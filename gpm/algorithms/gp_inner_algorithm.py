# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generative Planning (inner) Algorithm."""

from absl import logging
import numpy as np
import gin
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.td_loss import MSELoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import (TimeStep, Experience, LossInfo, namedtuple,
                                 AlgStep, StepType)
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork, QRNNNetwork, ParallelQNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import(losses, common, dist_utils, math_ops, spec_utils,
                      tensor_utils, value_ops)
from alf.utils.averager import WindowAverager, EMAverager
from alf.utils.losses import element_wise_squared_loss
from alf.utils.normalizers import AdaptiveNormalizer
from alf.utils.summary_utils import safe_mean_hist_summary


ActionType = Enum('ActionType', ('Discrete', 'Continuous', 'Mixed'))

GpActionState = namedtuple(
    "GpActionState", ["actor_network", "critic"], default_value=())

GpCriticState = namedtuple("GpCriticState", ["critics", "target_critics"])

GpState = namedtuple(
    "GpState", ["plan", "action", "actor", "critic"], default_value=())

Plan = namedtuple(
    "Plan", ["full_plan", "effective_plan_steps"], default_value=())

GpCriticInfo = namedtuple("GpCriticInfo", [
    "critics", "target_critic", "rewards", "critic_same_scale_mask",
    "critic_finer_scale_mask"
])

GpActorInfo = namedtuple(
    "GpActorInfo", ["actor_loss", "neg_entropy"], default_value=())

GpInfo = namedtuple(
    "GpInfo",
    [
        "p_commit",
        "natural_switch",
        "action",
        "actor",
        "critic",
        "alpha"
    ],
    default_value=())

GpLossInfo = namedtuple('GpLossInfo', ('actor', 'critic', 'alpha'))


def _set_target_entropy(name, target_entropy, flat_action_spec):
    """A helper function for computing the target entropy under different
    scenarios of ``target_entropy``.

    Args:
        name (str): the name of the algorithm that calls this function.
        target_entropy (float|Callable|None): If a floating value, it will return
            as it is. If a callable function, then it will be called on the action
            spec to calculate a target entropy. If ``None``, a default entropy will
            be calculated.
        flat_action_spec (list[TensorSpec]): a flattened list of action specs.
    """
    if target_entropy is None or callable(target_entropy):
        if target_entropy is None:
            target_entropy = dist_utils.calc_default_target_entropy
        target_entropy = np.sum(list(map(target_entropy, flat_action_spec)))
        logging.info("Target entropy is calculated for {}: {}.".format(
            name, target_entropy))
    else:
        logging.info("User-supplied target entropy for {}: {}".format(
            name, target_entropy))
    return target_entropy


@gin.configurable
class GpInnerAlgorithm(OffPolicyAlgorithm):
    r"""Generative Planning inner algorithm, which is used by Generative
    Planning agent.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 mini_batch_length,
                 action_to_world_conversion,
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 q_network_cls=QNetwork,
                 v_network_cls=QNetwork,
                 plan_length=1,
                 num_of_anchors=1,
                 min_target_commit_prob=0.9,
                 smooth_weight=1.,
                 mean_plan=False,
                 interp_method="linear",
                 replan_method="q",
                 critic_param_form="standard",
                 reward_discount_place="target",
                 use_target_reward_model=True,
                 gamma=0.99,
                 reward_weights=None,
                 use_entropy_reward=True,
                 use_parallel_network=True,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 reward_loss_ctor=MSELoss,
                 target_entropy=None,
                 initial_log_alpha=0.0,
                 init_epsilon=1.0,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 first_step_as_target=False,
                 averager_update_rate=1e-3,
                 actor_grad_scale=None,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 reward_optimizer=None,
                 alpha_optimizer=None,
                 epsilon_optimizer=None,
                 stochastic_replan=False,
                 shrink_threshold=0.0,
                 alg_render=False,
                 debug_summaries=False,
                 name="GpInnerAlgorithm"):
        """
        Args:
            num_of_anchors (int): number of anchor points for the plan
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Note that we don't need a discrete actor network
                because a discrete action can simply be sampled from the Q values.
            critic_param_form (str): 'standard' or 'decomposed'.
                - standard: conventional critic network
                - decompose: separate modeling and learning of reward and terminal
                    value. Reward is learned in a supervised way, and value is
                    learned in a value learning way.
                - dueling: separate Q(s, [a0, a1, ...]) = V(s) + A(s, [a0, a1, ...])
            mean_plan (str): if True, use mean plan value as 1) target 2) for
                actor training; otherwise, use full plan value (the last valid one)
            replan_method (str):
                - 'q': use q network for sampling
                - 'uniform': uniform sampling as exploration
            reward_discount_place (str):
                - 'target'
                - 'inference'
            use_target_reward_model (bool): whether use a slow target copy of reward
            critic_network_cls (Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.
            q_network (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            use_entropy_reward (bool): whether to include entropy as reward
            use_parallel_network (bool): whether to use parallel network for
                calculating critics.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            initial_log_alpha (float): initial value for variable ``log_alpha``.
            max_log_alpha (float|None): if not None, ``log_alpha`` will be
                capped at this value.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. For the mixed action type, discrete action and
                continuous action will have separate alphas and target entropies,
                so this argument can be a 2-element list/tuple, where the first
                is for discrete action and the second for continuous action.
                to constructor a prior actor. The output of the prior actor is
                the distribution of the next action. Two prior actors are implemented:
                ``alf.algorithms.prior_actor.SameActionPriorActor`` and
                ``alf.algorithms.prior_actor.UniformPriorActor``.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between
                ``[-dqda_clipping, dqda_clipping]``. Will not perform clipping if
                ``dqda_clipping == 0``.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            actor_grad_scale (bool): grad scaling applied for actor training,
                when computing the recurrent values for the action sequence
                sampled from actor
            stochastic_replan (bool): if False, use determinstic rule
                Q_new - Q_old > eps as the replanning criteria;
                otherwise construct a Categorical distribution and sample
                replanning signal from it:
                replanning ~ Cat(1/C, (Q_new - Q_old)/(C * eps))
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """

        self._num_critic_replicas = num_critic_replicas
        self._use_parallel_network = use_parallel_network
        self._interp_method = interp_method
        self._mean_plan = mean_plan
        self._actor_grad_scale = actor_grad_scale

        self._critic_param_form = critic_param_form
        self._replan_method = replan_method
        # whether apply the discount at training target or at inference
        self._reward_discount_place = reward_discount_place
        self._gamma = gamma
        self._use_target_reward_model = use_target_reward_model
        self._action_to_world_conversion = action_to_world_conversion

        self._action_res_dim = 0
        self._shrink_threshold = shrink_threshold

        self._stochastic_replan = stochastic_replan

        self._alg_render = alg_render

        self._first_step_as_target = first_step_as_target

        if len(action_spec) == 2:
            single_cnt_action_spec = action_spec[1]
            single_action_dim = single_cnt_action_spec.shape[0]
            # cnt_action_spec: single step action

            if single_cnt_action_spec.minimum.ndim == 0:
                # shared -1/1
                ones_vec = np.ones(single_action_dim)
                min_spec = np.tile(single_cnt_action_spec.minimum * ones_vec,
                                   num_of_anchors)
                max_spec = np.tile(single_cnt_action_spec.maximum * ones_vec,
                                   num_of_anchors)
            else:
                min_spec = np.tile(single_cnt_action_spec.minimum,
                                   num_of_anchors)
                max_spec = np.tile(single_cnt_action_spec.maximum,
                                   num_of_anchors)
            cnt_action_seq_spec = single_cnt_action_spec
            cnt_action_anchor_spec = BoundedTensorSpec(
                (single_action_dim * num_of_anchors, ),
                minimum=min_spec,
                maximum=max_spec)
            cnt_action_anchor_spec_w_res = BoundedTensorSpec(
                (single_action_dim * num_of_anchors + self._action_res_dim, ),
                minimum=np.concatenate(
                    (min_spec, -np.ones(self._action_res_dim)), axis=-1),
                maximum=np.concatenate(
                    (max_spec, np.ones(self._action_res_dim)), axis=-1))
            # action seq spec for full plan
            if single_cnt_action_spec.minimum.ndim == 0:
                ones_vec = np.ones(single_action_dim)
                seq_min_spec = np.tile(
                    single_cnt_action_spec.minimum * ones_vec, plan_length)
                seq_max_spec = np.tile(
                    single_cnt_action_spec.maximum * ones_vec, plan_length)
            else:
                seq_min_spec = np.tile(single_cnt_action_spec.minimum,
                                       plan_length)
                seq_max_spec = np.tile(single_cnt_action_spec.maximum,
                                       plan_length)

            cnt_action_seq_spec = BoundedTensorSpec(
                (plan_length * single_action_dim, ),
                minimum=seq_min_spec,
                maximum=seq_max_spec)

            extended_action_spec = (action_spec[0], cnt_action_seq_spec)

        # for entropy calculation
        self._action_anchor_spec = (action_spec[0], cnt_action_anchor_spec)
        self._single_cnt_action_spec = single_cnt_action_spec

        # for switch
        self._action_switch_spec = (BoundedTensorSpec(
            (), minimum=0, maximum=1), cnt_action_anchor_spec)
        self._cnt_action_anchor_spec = cnt_action_anchor_spec

        self._num_of_anchors = num_of_anchors
        self._plan_length = plan_length

        self._smooth_weight = smooth_weight

        self._mini_batch_length = mini_batch_length

        action_anchor_spec_w_res = (action_spec[0],
                                    cnt_action_anchor_spec_w_res)

        critic_networks, value_networks, actor_network, reward_dim = self._make_networks(
            observation_spec, action_anchor_spec_w_res, extended_action_spec,
            single_cnt_action_spec, actor_network_cls, critic_network_cls,
            q_network_cls, v_network_cls)

        self._use_entropy_reward = use_entropy_reward

        if reward_dim > 1:
            assert not use_entropy_reward, (
                "use_entropy_reward=True is not supported for multidimensional reward"
            )

        self._reward_weights = None
        if reward_weights:
            assert reward_dim > 1, (
                "reward_weights cannot be used for one dimensional reward")
            assert len(reward_weights) == reward_dim, (
                "Mismatch between len(reward_weights)=%s and reward_dim=%s" %
                (len(reward_weights), reward_dim))
            self._reward_weights = torch.tensor(
                reward_weights, dtype=torch.float32)

        def _init_log_alpha_replan():
            # use a large value to encourage commitment initially
            return nn.Parameter(torch.tensor(float(init_epsilon)))

        def _init_log_alpha():
            return nn.Parameter(torch.tensor(float(initial_log_alpha)))

        eps = _init_log_alpha_replan()
        log_alpha = _init_log_alpha()

        action_state_spec = GpActionState(
            actor_network=(actor_network.state_spec),
            critic=(critic_networks.state_spec))
        super().__init__(
            observation_spec,
            extended_action_spec,
            train_state_spec=GpState(
                action=action_state_spec,
                plan=Plan(
                    full_plan=cnt_action_seq_spec,
                    effective_plan_steps=action_spec[0]),
                actor=(critic_networks.state_spec),
                critic=GpCriticState(
                    critics=critic_networks.state_spec,
                    target_critics=critic_networks.state_spec)),
            predict_state_spec=GpState(
                action=action_state_spec,
                plan=Plan(
                    full_plan=cnt_action_seq_spec,
                    effective_plan_steps=action_spec[0])),
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        self._replan_averager = EMAverager(
            TensorSpec(()), averager_update_rate)

        self._abs_delta_q_averager = EMAverager(
            TensorSpec(()), averager_update_rate)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            if self._critic_param_form == "decompose":
                self.add_optimizer(reward_optimizer, [critic_networks])
            else:
                self.add_optimizer(critic_optimizer, [critic_networks])
            if value_networks is not None:
                self.add_optimizer(critic_optimizer, [value_networks])

        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, [log_alpha])

        self._log_alpha = log_alpha

        if epsilon_optimizer is not None:
            self.add_optimizer(epsilon_optimizer, [eps])

        self._eps = eps

        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(float(max_log_alpha))
        else:
            self._max_log_alpha = None

        self._actor_network = actor_network
        self._min_target_commit_prob = min_target_commit_prob


        self._critic_networks = critic_networks
        self._value_networks = value_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')
        if value_networks:
            self._target_value_networks = self._value_networks.copy(
                name='target_value_networks')
        else:
            self._target_value_networks = None

        if critic_loss_ctor is None:
            critic_loss_ctor = MultistepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._reward_losses = []
        for i in range(num_critic_replicas):
            self._reward_losses.append(
                reward_loss_ctor(
                    debug_summaries=debug_summaries,
                    name="reward_loss%d" % (i + 1)))

        self._target_entropy = _set_target_entropy(
            self.name, target_entropy,
            nest.flatten(self._single_cnt_action_spec))


        self._dqda_clipping = dqda_clipping

        if self._value_networks is None:
            self._update_target = common.get_target_updater(
                models=[self._critic_networks],
                target_models=[self._target_critic_networks],
                tau=target_update_tau,
                period=target_update_period)
        else:
            if self._use_target_reward_model or self._critic_param_form == "dueling":
                self._update_target = common.get_target_updater(
                    models=[self._critic_networks, self._value_networks],
                    target_models=[
                        self._target_critic_networks,
                        self._target_value_networks
                    ],
                    tau=target_update_tau,
                    period=target_update_period)
            else:
                # reward prediction module
                self._target_critic_networks = self._critic_networks
                self._update_target = common.get_target_updater(
                    models=[self._value_networks],
                    target_models=[self._target_value_networks],
                    tau=target_update_tau,
                    period=target_update_period)

    def _make_networks(self, observation_spec, anchor_action_spec, action_spec,
                       single_cnt_action_spec, continuous_actor_network_cls,
                       critic_network_cls, q_network_cls, v_network_cls):
        def _make_parallel(net):
            if self._use_parallel_network:
                nets = net.make_parallel(self._num_critic_replicas)
                #nets = net
            else:
                nets = alf.networks.NaiveParallelNetwork(
                    net, self._num_critic_replicas)
            return nets

        def _check_spec_equal(spec1, spec2):
            assert nest.flatten(spec1) == nest.flatten(spec2), (
                "Unmatched action specs: {} vs. {}".format(spec1, spec2))

        discrete_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_discrete
        ]
        continuous_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_continuous
        ]

        assert discrete_action_spec and continuous_action_spec, "need both"
        # When there are both continuous and discrete actions, we require
        # that acition_spec is a tuple/list ``(discrete, continuous)``.
        assert (isinstance(action_spec, (tuple, list))
                and len(action_spec) == 2), (
                    "In the mixed case, the action spec must be a tuple/list"
                    " (discrete_action_spec, continuous_action_spec)!")
        _check_spec_equal(action_spec[0], discrete_action_spec)
        _check_spec_equal(action_spec[1], continuous_action_spec)
        discrete_action_spec = action_spec[0]
        continuous_action_spec = action_spec[1]
        anchor_continuous_action_spec = anchor_action_spec[1]

        actor_network = None
        reward_dim = 1
        # for continuous_action_spec:
        assert continuous_actor_network_cls is not None, (
            "If there are continuous actions, then a ActorDistributionNetwork "
            "must be provided for sampling continuous actions!")
        # actor use anchor_continuous_action_spec
        actor_network = continuous_actor_network_cls(
            input_tensor_spec=observation_spec,
            action_spec=anchor_continuous_action_spec)

        # discrete_action_spec:
        assert reward_dim == 1, (
            "Discrete action is not supported for multidimensional reward")
        assert len(alf.nest.flatten(discrete_action_spec)) == 1, (
            "Only support at most one discrete action currently! "
            "Discrete action spec: {}".format(discrete_action_spec))
        assert q_network_cls is not None, (
            "If there exists a discrete action, then QNetwork must "
            "be provided!")

        additional_args = {"num_critic_replicas": self._num_critic_replicas}

        # critic network use continuous_action_spec
        if self._critic_param_form == "standard":
            # discrete_action_spec is equiv to _plan_length
            if self._replan_method == 'q':
                q_network = q_network_cls(
                    input_tensor_spec=(observation_spec,
                                       continuous_action_spec),
                    action_spec=discrete_action_spec,
                    **additional_args)
            else:
                q_network = q_network_cls(
                    input_tensor_spec=(observation_spec,
                                       continuous_action_spec),
                    action_spec=BoundedTensorSpec((), minimum=0, maximum=0))

            value_networks = None
        elif self._critic_param_form == "decompose":
            # 1) need to output num_unroll_steps (_plan_length)
            q_network = q_network_cls(
                input_tensor_spec=(observation_spec, continuous_action_spec),
                # can also use discrete_action_spec
                action_spec=discrete_action_spec,
                **additional_args)
            # 2) need to create a terminal value network, use Q-network
            # with scalar (single head) output; the input are
            # (obs, action_seqs) as it need to infer values of a future
            # state
            if self._replan_method == 'q':
                # in q replanning mode, we construct the terminal
                # value with multi-head
                value_network = q_network_cls(
                    input_tensor_spec=(observation_spec,
                                       continuous_action_spec),
                    action_spec=discrete_action_spec,
                    **additional_args)
            else:
                value_network = q_network_cls(
                    input_tensor_spec=(observation_spec,
                                       continuous_action_spec),
                    action_spec=BoundedTensorSpec((), minimum=0, maximum=0))

            value_networks = _make_parallel(value_network)

        elif self._critic_param_form == "dueling":
            # 1) need to output num_unroll_steps (_plan_length)
            # now the q-network is actually the adv network
            q_network = q_network_cls(
                input_tensor_spec=(observation_spec, continuous_action_spec),
                # can also use discrete_action_spec
                action_spec=discrete_action_spec,
                **additional_args)
            value_network = v_network_cls(
                input_tensor_spec=observation_spec,
                action_spec=BoundedTensorSpec((), minimum=0, maximum=0))
            value_networks = _make_parallel(value_network)

        critic_networks = _make_parallel(q_network)

        return critic_networks, value_networks, actor_network, reward_dim

    def _latent_to_action(self, latent_action):
        """latent to action mapping.
        """
        # action -> world-plan
        # [B, L*d] -> [B, L, d] -> [B, d, L]

        raw_action = latent_action.reshape(latent_action.shape[0],
                                           self._num_of_anchors, -1).permute(
                                               0, 2, 1)


        ego_actions = raw_action

        if self._interp_method == "linear":

            if self._action_to_world_conversion:
                padded_ego_actions = torch.cat(
                    (torch.zeros_like(ego_actions[..., 0:1]), ego_actions),
                    dim=-1)
                interp_size = self._plan_length + 1
            else:
                padded_ego_actions = ego_actions
                interp_size = self._plan_length

            # do interpolation here
            if interp_size == ego_actions.shape[-1]:
                ego_actions_interp = padded_ego_actions
            else:
                ego_actions_interp = torch.nn.functional.interpolate(
                    padded_ego_actions,
                    size=(interp_size),
                    mode='linear',
                    align_corners=True)

        elif self._interp_method == "quadratic":
            ego_actions = ego_actions.squeeze(2)
            ego_actions_interp = common.quad_interp(ego_actions,
                                                    self._plan_length)

        elif self._interp_method == "bezier":
            if self._action_to_world_conversion:
                # append zero control points to the beginning
                # [B, d, L] -> [B, d, L+1]
                padded_ego_actions = torch.cat(
                    (torch.zeros_like(ego_actions[..., 0:1]), ego_actions),
                    dim=-1)
                interp_size = self._plan_length + 1
            else:
                padded_ego_actions = ego_actions
                interp_size = self._plan_length

            ego_actions_interp = common.bezier_interp(padded_ego_actions,
                                                      interp_size)

        elif self._interp_method == "cubic":
            if self._action_to_world_conversion:
                # append zero control points to the beginning
                # [B, d, L] -> [B, d, L+1]
                padded_ego_actions = torch.cat(
                    (torch.zeros_like(ego_actions[..., 0:1]), ego_actions),
                    dim=-1)
                interp_size = self._plan_length + 1
            else:
                padded_ego_actions = ego_actions
                interp_size = self._plan_length

            ego_actions_interp = common.cubic_interp(padded_ego_actions,
                                                     interp_size)

        # remove ego if in proper mode, interpolation does not responsible
        # for this operation
        if self._action_to_world_conversion:
            # remove ego in this case
            ego_actions_interp = ego_actions_interp[..., 1:]

        # [B, d, L]
        action = ego_actions_interp.permute(0, 2, 1).reshape(
            latent_action.shape[0], -1)


        return action

    def _compute_multi_step_q_values(self, rewards, terminal_values):
        """
        Args:
            rewards:
                [B, replica, unroll_steps]
            terminal_values: the value of the state (terminal state of the
                multistep unroll)
                [B, replica, 1]
        Output:
            values:
                [B, replica, 1]
        """
        L = rewards.shape[-1]
        discount = (self._gamma
                    **torch.arange(L).float()).unsqueeze(0).unsqueeze(0)

        if self._reward_discount_place == 'inference':
            discounted_rewards = discount * rewards
            discounted_acc_rewards = torch.cumsum(discounted_rewards, dim=2)
        elif self._reward_discount_place == 'target':
            # otherwise, the network output is discounted and accumulated rewards
            discounted_acc_rewards = rewards

        if self._replan_method == 'q':
            # [B, replica, unroll_steps]
            value = discounted_acc_rewards + self._gamma * discount * terminal_values
        else:
            discounted_acc_rewards[..., -1:] = discounted_acc_rewards[
                ..., -1:] + self._gamma * discount[..., -1:] * terminal_values

            value = (discounted_acc_rewards).sum(dim=2, keepdims=True)
        return value

    def _get_epsilon(self, detach):
        epsilon = nn.functional.softplus(self._eps)
        if detach:
            epsilon = epsilon.detach()
        return epsilon

    def _predict_action(self,
                        observation,
                        prev_plan,
                        state: GpActionState,
                        unroll_flag,
                        epsilon_greedy=None):
        """The reason why we want to do action sampling inside this function
        instead of outside is that for the mixed case, once a continuous action
        is sampled here, we should pair it with the discrete action sampled from
        the Q value. If we just return two distributions and sample outside, then
        the actions will not match.

        Note that the output action is full, not the sparse anchors.

        prev_plan: if None, compute critic values for both current plan
            and the previous plan, which is used in rollout and predict.
            In train, we should set it to None to avoid unnecessary computation.
        """

        new_state = GpActionState()

        continuous_action_latent, log_pi_cont, actor_network_state = self._actor_network(
            observation,
            partial_mean=None,
            epsilon_greedy=epsilon_greedy,
            state=state.actor_network)
        continuous_action = self._latent_to_action(continuous_action_latent)

        new_state = new_state._replace(actor_network=actor_network_state)

        q_values = None
        discrete_action_dist = None
        natural_switch = None

        if prev_plan != None:
            # perform joint forwarding for speedup
            q_values_joint, _, _, critic_state = self._compute_critics(
                self._critic_networks,
                torch.cat([observation, observation], dim=0), [
                    None,
                    torch.cat([prev_plan.full_plan, continuous_action], dim=0)
                ], state.critic, self._value_networks)
            q_values_split = torch.split(q_values_joint, observation.shape[0])
            q_values_prev = q_values_split[0]
            q_values = q_values_split[1]
        else:
            q_values, _, _, critic_state = self._compute_critics(
                self._critic_networks, observation, [None, continuous_action],
                state.critic, self._value_networks)

        new_state = new_state._replace(critic=critic_state)

        if self._replan_method == 'q':
            epsilon = self._get_epsilon(detach=True)

            if prev_plan != None:
                # select according to effective plan length
                # effective_plan_steps - 1 as index
                # initially, effective_plan_steps might be zero
                # due to state reset, use a mod operation for correction

                # if prev_plan.effective_plan_steps is zero, that means the
                # previous full_plan has been fully executed, and we need to
                # switch naturally. The switch decision in this case
                # should not be counted into the switch frequency

                natural_switch = (prev_plan.effective_plan_steps == 0)

                mod_step = (prev_plan.effective_plan_steps - 1 +
                            self._plan_length) % self._plan_length

                q_values_prev_for_sel = self._select_q_value(
                    mod_step, q_values_prev)

                q_values_for_sel = self._select_q_value(mod_step, q_values)


                # [B, L]
                q_values_for_sel = q_values_for_sel.min(dim=1)[0]
                q_values_prev_for_sel = q_values_prev_for_sel.min(dim=1)[0]

                q_values = q_values.min(dim=1)[0]

                if self._mean_plan:
                    q_values = q_values.mean(-1)
                else:
                    q_values = q_values[..., -1]
                # q_values = q_values[..., 0]

            else:
                q_values_for_sel = q_values[..., -1]

                q_values_for_sel = q_values_for_sel.min(dim=1)[0]
                q_values_prev_for_sel = q_values_for_sel.clone().detach(
                ) - 1e10

                # [B, L]
                q_values = q_values.min(dim=1)[0]

                if self._mean_plan:
                    q_values = q_values.mean(-1)
                else:
                    q_values = q_values[..., -1]



            delta = q_values_for_sel - q_values_prev_for_sel


            delta_q_values = torch.stack(
                [torch.ones_like(q_values) * epsilon, delta], -1)

            if unroll_flag and prev_plan != None:
                self._abs_delta_q_averager.update(torch.abs(delta))

            avg_q = torch.clamp(self._abs_delta_q_averager.get(), min=1e-5)

            # only do summary in the case when previous plan is not None
            if prev_plan != None and self._debug_summaries and alf.summary.should_record_summaries(
            ):
                with alf.summary.scope(self._name):
                    alf.summary.histogram("Q_new_old_delta", delta)
                    alf.summary.scalar("Q_new_old_delta_val", delta.mean())
                    alf.summary.scalar("plan_switch_epsilon", epsilon)

            logits = delta_q_values

            def get_commit_switch_action(delta, epsilon, natural_switch):
                # commit_switch_action: 0 commit; 1 switch
                commit_switch_action = torch.zeros(delta.shape[0]).long()

                switch_mask = delta > epsilon

                # include natural switch
                if natural_switch is not None:
                    switch_mask = switch_mask | natural_switch

                commit_switch_action[switch_mask] = 1
                return commit_switch_action.detach()

            def get_stochastic_switch_action(switch_mask, natural_switch):
                # commit_switch_action: 0 commit; 1 switch
                commit_switch_action = torch.zeros(delta.shape[0]).long()

                switch_mask = switch_mask.bool()
                # include natural switch
                if natural_switch is not None:
                    switch_mask = switch_mask | natural_switch

                commit_switch_action[switch_mask] = 1
                return commit_switch_action.detach()

            if not self._stochastic_replan:
                discrete_action = get_commit_switch_action(
                    delta, epsilon, natural_switch)
            else:
                discrete_action_dist = td.Categorical(logits=logits)

                greedy_action = (delta > epsilon)

                discrete_action = dist_utils.epsilon_greedy_sample(
                    discrete_action_dist, eps=epsilon_greedy)


                if self._debug_summaries and alf.summary.should_record_summaries(
                ):
                    with alf.summary.scope(self._name):
                        alf.summary.scalar("prob0_commit",
                                           discrete_action_dist.probs[0, 0])
                        alf.summary.scalar("prob1_switch",
                                           discrete_action_dist.probs[0, 1])
                        alf.summary.scalar(
                            "average_q_for_replan_normalization", avg_q)

        elif self._replan_method == "uniform":
            discrete_action = torch.randint(0, self._plan_length,
                                            (observation.shape[0], ))
            # dummy dist
            discrete_action_dist = td.Categorical(
                logits=torch.ones(observation.shape[0]))

        if self._replan_method == 'q':
            # convert to switch action and use natural switch mask at task
            # for using only the valid ones
            p_commit = 1 - discrete_action.float()

            # when update length, should do when there is a natural switch
            # length
            if self._min_target_commit_prob >= 1:
                # only need to update averager when:
                # 1) unroll and
                # 2) there is a natural switch OR decision swich
                if unroll_flag and (any(natural_switch)
                                    or any(discrete_action)):
                    # only use the valid part to update averager
                    steps_taken = self._plan_length - prev_plan.effective_plan_steps
                    # natural switch OR decision switch
                    sel = natural_switch | discrete_action.bool()
                    p_commit_valid = steps_taken[sel].float()
                    # p_commit_valid = torch.clamp(p_commit_valid, min=1e-4)
                    self._replan_averager.update(p_commit_valid)
            else:

                # # 1) natural_switch should not ne None: only in unroll case
                # # only use valid data to update statistics
                if unroll_flag and (any(natural_switch)
                                    or any(discrete_action)):
                    # only use the valid part to update averager
                    p_commit_valid = p_commit[~natural_switch]
                    p_commit_valid = torch.clamp(p_commit_valid, min=1e-4)
                    self._replan_averager.update(p_commit_valid)
                # p_commit = self._replan_averager.get()

        elif self._replan_method == "uniform":
            log_pi_disc = torch.zeros((discrete_action.shape[0]))

        action = type(self._action_spec)((discrete_action, continuous_action))
        action_w_latent = type(self._action_spec)((discrete_action,
                                                   continuous_action_latent))
        log_pi = log_pi_cont

        return log_pi, action, action_w_latent, q_values, discrete_action_dist, new_state, natural_switch, q_values_prev_for_sel, q_values_for_sel, p_commit

    def predict_step(self,
                     time_step: TimeStep,
                     state: GpState,
                     epsilon_greedy=1.0,
                     pose=None,
                     prev_traj=None):

        previous_plan = state.plan

        action_dist, action, _, q_values_policy, _, action_state, _, q_values_prev_for_sel, q_values_for_sel, _ = self._predict_action(
            time_step.observation,
            state=state.action,
            prev_plan=previous_plan,
            unroll_flag=False,
            epsilon_greedy=epsilon_greedy)

        steps = torch.full_like(action[0], self._plan_length - 1)
        new_plan = Plan(full_plan=action[1], effective_plan_steps=steps + 1)

        def _unflatten_action_seq(continuous_action):
            continuous_action = continuous_action.reshape(
                continuous_action.shape[0], self._plan_length, -1).permute(
                    0, 2, 1)
            return continuous_action

        continuous_action = _unflatten_action_seq(action[1])

        Q_ind = -1

        action_spec = self._action_spec[1]

        delta_x = (action_spec.maximum[0] - action_spec.minimum[0]) / 10.0
        delta_y = (action_spec.maximum[0] - action_spec.minimum[0]) / 10.0
        x_min = action_spec.minimum[0] - delta_x
        x_max = action_spec.maximum[0] + delta_x

        y_min = action_spec.minimum[0]
        y_max = action_spec.maximum[0]

        if self._alg_render:
            with alf.summary.scope("GPM"):
                img_size = 256
                q_values_for_sel = q_values_for_sel.cpu().numpy()

                q_val_all = q_values_policy.cpu().numpy()
                eps = self._get_epsilon(detach=True).cpu().numpy()

                plan_new = alf.summary.render.render_traj_set(
                    name="plan_new",
                    data=continuous_action,
                    q_val=q_values_for_sel,
                    q_val_full_traj=q_val_all,
                    #eps=eps,
                    x_range=[0, self._plan_length],
                    y_range=[y_min, y_max],
                    img_height=img_size,
                    img_width=img_size)

                q_values_prev_for_sel = q_values_prev_for_sel.cpu().numpy()

                plan_old = previous_plan.full_plan.unsqueeze(1)

                plan_old = _unflatten_action_seq(plan_old)

                effective_plan_steps = previous_plan.effective_plan_steps.cpu(
                ).numpy()

                # replan, use full steps
                if action[0]:
                    mod_step = torch.full((1, ), self._plan_length)
                else:
                    mod_step = (effective_plan_steps - 1 +
                                self._plan_length) % self._plan_length + 1

                steps_for_sel = (effective_plan_steps - 1 +
                                 self._plan_length) % self._plan_length + 1
                plan_old = alf.summary.render.render_traj_set(
                    name="plan_old",
                    data=plan_old,
                    q_val=q_values_prev_for_sel,
                    eps=eps,
                    effective_plan_steps=steps_for_sel,
                    x_range=[0, self._plan_length],
                    y_range=[y_min, y_max],
                    img_height=img_size,
                    img_width=img_size)

                effective_steps_image = alf.summary.render.render_bar(
                    y_range=[0, self._plan_length],
                    name="effective_steps",
                    data=mod_step)

            info = dict(
                plan_new=plan_new,
                plan_old=plan_old,
                effective_steps_image=effective_steps_image)
        else:
            info = {}

        self._info = info

        new_state = GpState(action=action_state, plan=new_plan)
        return AlgStep(
            output=(*action, q_values_prev_for_sel), state=new_state)


    def rollout_step(self, time_step: TimeStep, state: GpState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """

        previous_plan = state.plan
        detached_observation = common.detach(time_step.observation)
        log_pi, action, _, _, _, action_state, natural_switch, _, _, p_commit = self._predict_action(
            detached_observation,
            prev_plan=previous_plan,
            state=state.action,
            unroll_flag=True,  # used to update statistics
            epsilon_greedy=1.0)


        # set steps to full, since we want to commit by default
        # using index representation [0, K-1], later move to [1, K]
        # shen setting values for state
        steps = torch.full_like(action[0], self._plan_length - 1)
        new_plan = Plan(full_plan=action[1], effective_plan_steps=steps + 1)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, time_step.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, time_step.observation, action,
                state.critic.target_critics)
            critic_state = GpCriticState(
                critics=critics_state, target_critics=target_critics_state)
            actor_state = ()
        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = GpState(
            action=action_state,
            actor=actor_state,
            critic=critic_state,
            plan=new_plan)
        return AlgStep(
            output=(*action, natural_switch),
            state=new_state,
            info=GpInfo(p_commit=p_commit, natural_switch=natural_switch))

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         value_net=None,
                         grad_scale=None,
                         detach_reward=False):
        """
        Args:
            detach_reward (bool): only has effect when critic_pram_form is
                ''standard''
        """

        org_observation = observation

        if isinstance(
                    critic_net, alf.networks.NaiveParallelNetwork) or \
            isinstance(critic_net, alf.networks.AutoRegressiveQNetwork):
            # [B, L, d]
            L = self._plan_length
            cnt_action = action[1]  #.unsqueeze(-1)
            observation = (observation, cnt_action)

        # discrete/mixed: critics shape [B, replicas, num_actions]
        # continuous: critics shape [B, replicas]
        value = None
        reward = None
        if self._critic_param_form == "standard":
            # [B, replica, 1]
            critics, _ = critic_net(
                observation, state=(), grad_scale=grad_scale)
            rewards = None
        elif self._critic_param_form == "decomposed":
            #[B, replica, unroll_steps]
            rewards, critics_state = critic_net(
                observation, state=critics_state)

            rewards_for_value = rewards.mean(dim=1, keepdim=True)
            if detach_reward:
                rewards_for_value = rewards_for_value.detach()

            #[B, 1]
            value, _ = value_net(observation, state=critics_state)
            critics = self._compute_multi_step_q_values(
                rewards_for_value, value)
        elif self._critic_param_form == "dueling":
            #[B, replica, unroll_steps]
            advs, critics_state = critic_net(observation, state=critics_state)

            advs_for_value = advs - advs.max(dim=-1, keepdim=True)[0]
            # advs_for_value = advs

            #[B, 1]
            # only observation
            value, _ = value_net(org_observation, state=critics_state)
            critics = value + advs_for_value

            rewards = None

        return critics, rewards, value, critics_state

    def _smoothness_cost(self, actions, prev_traj, pose):

        # [B, L, d] -> [B, d, L]

        actions = actions.reshape(actions.shape[0], self._plan_length,
                                  -1).permute(0, 2, 1)

        concat_actions = actions

        diff = concat_actions[..., 1:] - concat_actions[..., :-1]
        diff2 = diff[..., 1:] - diff[..., :-1]
        diff3 = diff2[..., 1:] - diff2[..., :-1]

        loss = (diff3.abs()).max(-1)[0].mean(-1)

        if alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("loss/smoothness", loss.mean())
        return loss

    def _actor_train_step(self, exp: Experience, state, action, critics,
                          log_pi, pose, prev_traj):
        neg_entropy = sum(nest.flatten(log_pi))
        # use the critics computed during action prediction for Mixed type
        # ``critics``` is already after min over replicas
        critics_state = ()

        continuous_log_pi = log_pi
        cont_alpha = torch.exp(self._log_alpha).detach()

        q_value = critics.min(dim=1)[0]
        q_value = q_value.sum(-1)
        dqda = nest_utils.grad(action, q_value.sum(), retain_graph=False)


        def actor_loss_fn(dqda, action):
            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope(self._name):
                    alf.summary.scalar("dqda_norm",
                                       torch.norm(dqda, dim=-1).mean())
                    alf.summary.scalar("dqda_abs_max", dqda.abs().max())
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)

            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))

        full_actor_loss = actor_loss + cont_alpha * continuous_log_pi

        if self._smooth_weight > 0:
            smooth_loss = self._smoothness_cost(action, prev_traj, pose)
            full_actor_loss = full_actor_loss + self._smooth_weight * smooth_loss

        actor_info = LossInfo(
            loss=full_actor_loss,
            extra=GpActorInfo(actor_loss=actor_loss, neg_entropy=neg_entropy))

        return critics_state, actor_info

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action (Tensor): discrete actions with shape ``[batch_size]``.
            q_values (Tensor): Q values with shape ``[batch_size, replicas, num_actions]``.
        Returns:
            Tensor: selected Q values with shape ``[batch_size, replicas]``.
        """
        # action shape: [batch_size] -> [batch_size, n, 1]
        action = action.view(q_values.shape[0], 1, -1).expand(
            -1, q_values.shape[1], -1).long()
        return q_values.gather(-1, action).squeeze(-1)
        # return q_values.squeeze(-1)

    def _action_to_sequence(self, action, step_type, method="zero"):
        """Reorganize ``action`` to sequences
        Args:
            action (Tensor): discrete actions with shape ``[L*B, d]``, where d is
            the dimensionality of the single atomic action.
        Returns:
            - Tensor: ``[L*B, K*d]``, where L is the action sequence length
            - Tensor: ``[L*B, 1]``
        """
        K = self._num_of_anchors
        d = action.shape[1]

        valid_mask = (step_type != StepType.LAST)

        valid_mask = valid_mask.reshape(self._mini_batch_length, -1)

        # [L*B, d, 1]
        single_action = action.reshape(action.shape[0], d, 1)

        # recover temporal structure
        # [L, B, d, 1]
        tc_actions = single_action.reshape(self._mini_batch_length, -1,
                                           *single_action.shape[1:])

        L = tc_actions.shape[0]
        B = tc_actions.shape[1]

        assert K == L - 1, "only K = L - 1 is supported currently"

        # init processed actions, with all zeros
        # [L, B, d, K]
        proc_actions = torch.randn(*tc_actions.shape[:-1], K)

        discrete_actions = torch.zeros(L, B, 1)

        for t in range(K):
            # [L-t, B, d, 1]
            ai = tc_actions[t:-1]
            ai_permute = ai.permute(3, 1, 2, 0)

            proc_actions[t:t + 1, :, :, 0:ai_permute.shape[-1]] = ai_permute

            discrete_actions[t, :] = max(L - t - 2, 0)

        proc_actions = proc_actions.permute(0, 1, 3, 2).reshape(-1, K * d)

        return discrete_actions.reshape(-1, 1), proc_actions

    def _critic_train_step(self, exp: Experience, state: GpCriticState, action,
                           log_pi, action_terminal, log_pi_terminal,
                           state_action):
        """
        state_action: for action generation
        """

        exp_action_seq = exp.rollout_info.action[1]
        # [B, L, d] -> [B, L*d]
        exp_action_seq = exp_action_seq.reshape(exp_action_seq.shape[0], -1)
        exp_discrete_actions = exp.action[0]

        # no need to do this, the sampled discrete action is already correct
        sampled_unroll_step = exp.rollout_info.sampled_unroll_step

        # -1: steps to index
        exp_discrete_actions = sampled_unroll_step - 1

        terminal_observation = exp.rollout_info.terminal_observation

        action_for_critic = (exp_discrete_actions, exp_action_seq)
        critics, rewards, _, critics_state = self._compute_critics(
            self._critic_networks,
            exp.observation,
            action_for_critic,
            state.critics,
            self._value_networks,
            detach_reward=True)
        if rewards is None:
            rewards = critics.detach()  # dummy

        # need to generate new action
        # resample actions and compute log_pi for target critic
        with torch.no_grad():
            action = action_terminal.detach()
            log_pi = log_pi_terminal.detach()
            target_critics, _, target_value, target_critics_state = self._compute_critics(
                self._target_critic_networks, terminal_observation,
                (None, action), state.target_critics,
                self._target_value_networks)

            target_critics = target_critics.min(dim=1)[0]

        positions_full = torch.arange(0, self._plan_length)
        range_ind = positions_full.unsqueeze(0).expand(
            exp_discrete_actions.shape[0], self._plan_length)
        critic_same_scale_mask = (
            range_ind == exp_discrete_actions.unsqueeze(1))
        critic_same_scale_mask = critic_same_scale_mask.float()

        critic_finer_scale_mask = (range_ind <
                                   exp_discrete_actions.unsqueeze(1))
        critic_finer_scale_mask = critic_finer_scale_mask.float()

        if self._replan_method == "q":
            if not self._first_step_as_target:
                if self._mean_plan:
                    target_critics = torch.mean(target_critics, dim=-1)
                else:
                    target_critics = target_critics[..., -1]
            else:
                # always use 0
                target_critics = target_critics[..., 0]

        else:
            critics = critics.squeeze(-1)
            target_critics = target_critics.squeeze(-1)

        target_critic = target_critics.reshape(exp.reward.shape)
        if self._use_entropy_reward and self._critic_param_form != "dueling":
            entropy_reward = -torch.exp(self._log_alpha) * log_pi
            target_critic = target_critic + entropy_reward

        target_critic = target_critic.detach()

        state = GpCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = GpCriticInfo(
            critics=critics,
            target_critic=target_critic,
            rewards=rewards,
            critic_same_scale_mask=critic_same_scale_mask,
            critic_finer_scale_mask=critic_finer_scale_mask)

        return state, info

    def _alpha_train_step(self, log_pi, p_commit, natural_switch):

        p_min_commit_target = self._min_target_commit_prob

        epsilon = self._get_epsilon(detach=False)

        eps_loss = epsilon * (p_commit - p_min_commit_target).detach()

        alpha_loss = self._log_alpha * (
            -log_pi - self._target_entropy).detach()

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/p_commit", p_commit.mean())
                alf.summary.scalar("alpha/epsilon_loss", eps_loss.mean())
                alf.summary.scalar("alpha/alpha_loss", alpha_loss.mean())
                alf.summary.scalar("ent/continuous", -log_pi.mean())

        alpha_loss = eps_loss + alpha_loss

        return alpha_loss

    def train_step(self, exp: Experience, state: GpState, pose, prev_traj):
        # We detach exp.observation here so that in the case that exp.observation
        # is calculated by some other trainable module, the training of that
        # module will not be affected by the gradient back-propagated from the
        # actor. However, the gradient from critic will still affect the training
        # of that module.

        previous_plan = state.plan

        terminal_observation = exp.rollout_info.terminal_observation

        new_action_state = GpActionState()

        continuous_action_latent, log_pi_cont, actor_network_state = self._actor_network(
            common.detach(
                torch.cat([exp.observation, terminal_observation], dim=0)),
            partial_mean=None,
            epsilon_greedy=None,
            state=state.action.actor_network)
        continuous_action = self._latent_to_action(continuous_action_latent)

        b_size = exp.observation.shape[0]
        log_pi_split = torch.split(log_pi_cont, b_size)
        action_split = torch.split(continuous_action, b_size)

        log_pi = log_pi_split[0]
        action = action_split[0]
        log_pi_terminal = log_pi_split[1]
        action_terminal = action_split[1]

        new_action_state = new_action_state._replace(
            actor_network=actor_network_state)

        # for actor training, we need to take min later
        q_values, _, _, critic_state = self._compute_critics(
            self._critic_networks,
            exp.observation, [None, action],
            state.critic,
            self._value_networks,
            grad_scale=self._actor_grad_scale)

        if self._mean_plan:
            critics = q_values.mean(dim=-1)
        else:
            critics = q_values

        new_action_state = new_action_state._replace(critic=critic_state)

        actor_state, actor_loss = self._actor_train_step(
            exp, state.actor, action, critics, log_pi, pose, prev_traj)
        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action, log_pi, action_terminal,
            log_pi_terminal, state.action)

        # use averager to get the new rollout statistics
        p_commit = self._replan_averager.get()

        natural_switch = (previous_plan.effective_plan_steps == 0)
        alpha_loss = self._alpha_train_step(log_pi, p_commit, natural_switch)

        state = GpState(
            action=new_action_state, actor=actor_state, critic=critic_state)
        info = GpInfo(
            action=action,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss
        )
        return AlgStep(action, state, info)

    def after_update(self, experience, train_info: GpInfo):
        self._update_target()
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

    def calc_loss(self, experience, train_info: GpInfo):
        critic_loss = self._calc_critic_loss(experience, train_info)
        alpha_loss = train_info.alpha
        actor_loss = train_info.actor

        pose = None

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())

        return LossInfo(
            loss=math_ops.add_ignore_empty(actor_loss.loss,
                                           critic_loss.loss + alpha_loss),
            priority=critic_loss.priority,
            extra=GpLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss))

    def _calc_critic_loss(self, experience, train_info: GpInfo):
        critic_info = train_info.critic

        # [T, B, unroll_steps]
        exp_reward_seq = experience.rollout_info.reward
        mask = experience.rollout_info.mask
        valid_mask = experience.rollout_info.valid_mask
        last_step_mask = experience.rollout_info.last_step_mask

        discount = experience.rollout_info.discount

        # [T, B, replica, unroll_steps]
        predicted_rewards = critic_info.rewards
        critic_same_scale_mask = critic_info.critic_same_scale_mask
        critic_finer_scale_mask = critic_info.critic_finer_scale_mask

        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(value=critic_info.critics[:, :, i, ...],
                  terminal_value=critic_info.target_critic,
                  discount=discount,
                  rewards=exp_reward_seq,
                  mask=mask,
                  last_step_mask=last_step_mask,
                  critic_same_scale_mask=critic_same_scale_mask,
                  critic_finer_scale_mask=critic_finer_scale_mask).loss)
        critic_loss = math_ops.add_n(critic_losses)

        # apply mask, for cross episode case, predict zero
        masked_exp_reward_seq = valid_mask * exp_reward_seq
        if self._critic_param_form == "decompose":

            L = masked_exp_reward_seq.shape[-1] - 1
            if self._reward_discount_place == 'inference':
                reward_target = masked_exp_reward_seq[..., 1:]
            elif self._reward_discount_place == 'target':
                with torch.no_grad():
                    discount = (
                        self._gamma
                        **torch.arange(L).float()).unsqueeze(0).unsqueeze(0)
                    discounted_rewards = discount * masked_exp_reward_seq[...,
                                                                          1:]
                    discounted_acc_rewards = torch.cumsum(
                        discounted_rewards, dim=2)
                    reward_target = discounted_acc_rewards

            reward_losses = []
            # need to use valid_mask to avoid masking out the final step
            for i, l in enumerate(self._reward_losses):
                reward_losses.append(
                    l(pred=predicted_rewards[:, :, i, ...],
                      target=reward_target).loss)

            reward_loss = math_ops.add_n(reward_losses)

            # remove invalid data loss
            # note: should use valid_mask[..., 1]
            reward_loss = reward_loss * valid_mask[..., 1]

            critic_loss = critic_loss + reward_loss

        if (experience.batch_info != ()
                and experience.batch_info.importance_weights != ()):
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_critic_networks', '_target_value_networks',
            '_replan_averager'
        ]


@gin.configurable
class MultistepTDLoss(nn.Module):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
                 normalize_target=False,
                 debug_summaries=False,
                 name="MultistepTDLoss"):
        r"""Create a MultistepTDLoss object.
        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_target (bool): whether to normalize target.
                Note that the effect of this is to change the loss. The critic
                value itself is not normalized.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._lambda = td_lambda
        self._debug_summaries = debug_summaries
        self._normalize_target = normalize_target
        self._target_normalizer = None

    def forward(self, value, terminal_value, discount, rewards, mask,
                last_step_mask, critic_same_scale_mask,
                critic_finer_scale_mask):
        """Cacluate the loss.
        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.
        T: mini_batch_length
        B: batch size
        L: unroll length
        Args:
            value (torch.Tensor): [T=1, B]
                the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            terminal_value (torch.Tensor): [T=1, B]
            terminal_discount (torch.Tensor): [T=1, B, 1]
            rewards (torch.Tensor): [T=1, B, L]
            mask (torch.Tensor): [T=1, B, L]
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """

        L = rewards.shape[-1] - 1

        returns = value_ops.discounted_return_multistep_unroll(
            terminal_value=terminal_value,
            discount=discount * self._gamma,
            rewards=rewards,
            mask=mask,
            last_step_mask=last_step_mask)

        if self._normalize_target:
            if self._target_normalizer is None:
                self._target_normalizer = AdaptiveNormalizer(
                    alf.TensorSpec(value.shape[2:]),
                    auto_update=False,
                    debug_summaries=self._debug_summaries,
                    name=self._name + ".target_normalizer")

            self._target_normalizer.update(returns)
            returns = self._target_normalizer.normalize(returns)
            value = self._target_normalizer.normalize(value)

        if value.ndim == 3 and returns.ndim == 2:
            returns = returns.unsqueeze(-1)

        if self._debug_summaries and alf.summary.should_record_summaries():

            mask_max_values, mask_max_indices = torch.max(
                critic_same_scale_mask, dim=-1)

            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r))
                    safe_mean_hist_summary('values' + suffix, v)
                    safe_mean_hist_summary('returns' + suffix, r)
                    safe_mean_hist_summary("td_error" + suffix, td)

                valid_mask = mask[..., 0]
                for plan_len_l in range(L):
                    sel_mask = (mask_max_indices == plan_len_l) & valid_mask
                    ind = torch.nonzero(sel_mask.squeeze(0)).squeeze(-1)
                    if ind.numel() > 0:
                        value_correct_scale = value.gather(
                            -1, mask_max_indices.unsqueeze(-1))
                        sel_v = value_correct_scale.index_select(1, ind)
                        sel_return = returns.index_select(1, ind)
                        _summarize(sel_v, sel_return, sel_return - sel_v,
                                   '_' + str(plan_len_l))

        loss = self._td_error_loss_fn(returns.detach(), value)

        # only the finer scales with larger target value
        valid_finer_scale = critic_finer_scale_mask * (returns >
                                                       value).detach()

        loss = loss * critic_same_scale_mask
        mask = mask[..., 0]

        if loss.ndim == 3:
            loss = loss.sum(dim=2)

        loss = loss * mask

        return LossInfo(loss=loss, extra=loss)