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

import gin
import torch
import copy
import math
import numpy as np

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import RewardNormalizer
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.data_structures import TimeStep, Experience, namedtuple, AlgStep
from alf.data_structures import make_experience, LossInfo
from alf.nest import nest
from alf.nest.utils import convert_device
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.conditional_ops import conditional_update
from alf.utils import common, summary_utils, tensor_utils

from .gp_inner_algorithm import GpInnerAlgorithm

# full plan: the plan at full traj length, and reduced by 1 each time step
# effective plan: the plan cut off by the predicted commitment steps, and reduced
# by 1 at each time step
# effective plan:
GenerativePlanningState = namedtuple(
    "GenerativePlanningState",
    [
        "rl",
        "action",
        "switch_flag",
        "effective_plan_steps",  # remaining number of steps for the effective plan
        "full_plan_steps",  # remaining number of steps for the full plan
        "k",  # executed step counts
        "rl_discount",
        "rl_reward",
        "sample_rewards",
        "repr"
    ],
    default_value=())

GenerativePlanningInfo = namedtuple(
    'GenerativePlanningInfo',
    [
        # actual actions taken in the next unroll_steps + 1 steps
        # [B, unroll_steps, ...]
        'action',

        # [B, unroll_steps, ...]
        'reward',

        # The flag to indicate whether to include this target into loss
        # [B, unroll_steps]
        'mask',
        'valid_mask',
        'last_step_mask',

        # # nest for targets
        # # [B, unroll_steps, ...]
        # 'target',

        # [B, unroll_steps]
        'terminal_observation',

        # [B, unroll_steps]
        'discount',
        'sampled_unroll_step',

        # for replan threshold adjustment
        "p_commit",
        "natural_switch"
    ])

Plan = namedtuple(
    "Plan", ["full_plan", "effective_plan_steps"], default_value=())

ReplanInfo = namedtuple('ReplanInfo', ["p_commit", "natural_switch"])


@gin.configurable
class GenerativePlanningAgent(OffPolicyAlgorithm):
    """Generative Planning Method <https://arxiv.org/pdf/2201.09765.pdf>`_.
    """

    def __init__(
            self,
            observation_spec,
            action_spec,
            num_of_anchors,
            plan_length,
            mini_batch_length,
            num_unroll_steps,
            action_to_world_conversion,
            reward_clip_value=-1,
            target_fields='',
            env=None,
            config: TrainerConfig = None,
            param_method="xy",
            representation_learner_cls=None,
            reward_normalizer_ctor=None,
            gamma=0.99,
            optimizer=None,
            debug_summaries=False,
            alg_render=False,
            name="GenerativePlanningAgent"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                only be continuous actions for now.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            K (int): the maiximal repeating times for an action.

            action_to_world_conversion (bool): whether need action to world
                conversion.
                - low-level action: should be False
                - PID: True
            param_method (str): parametrization method
                - "xy"
                - "theta"
            representation_learner_cls (type): The algorithm class for learning
                the representation. If provided, the constructed learner will
                calculate the representation from the original observation as
                the observation for downstream algorithms such as ``rl_algorithm``.
                We assume that the representation is trained by ``rl_algorithm``.
            reward_normalizer_ctor (Callable): if not None, it must be
                ``RewardNormalizer`` and environment rewards will be normalized
                for training.
            gamma (float): the reward discount to be applied when accumulating
                ``k`` steps' rewards for a repeated action. Note that this value
                should be equal to the gamma used by the critic loss for target
                values.
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this agent.
        """
        assert action_spec.is_continuous, (
            "Only support continuous actions for now!")

        rl_observation_spec = observation_spec

        repr_learner = None
        if representation_learner_cls is not None:
            repr_learner = representation_learner_cls(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            rl_observation_spec = repr_learner.output_spec

        self._rl_action_spec = (BoundedTensorSpec(
            shape=(), dtype='int64', maximum=plan_length - 1), action_spec)
        rl = GpInnerAlgorithm(
            observation_spec=rl_observation_spec,
            action_spec=self._rl_action_spec,
            debug_summaries=debug_summaries)

        single_action_dim = action_spec.shape[0]

        if action_spec.minimum.ndim == 0:
            # shared -1/1
            ones_vec = np.ones(single_action_dim)
            single_min = action_spec.minimum * ones_vec
            single_max = action_spec.maximum * ones_vec
            min_spec = np.tile(single_min, plan_length)
            max_spec = np.tile(single_max, plan_length)
        else:
            single_min = action_spec.minimum
            single_max = action_spec.maximum
            min_spec = np.tile(single_min, plan_length)
            max_spec = np.tile(single_max, plan_length)

        ext_action_spec = BoundedTensorSpec(
            (single_action_dim * plan_length, ),
            minimum=min_spec,
            maximum=max_spec)

        pad_action_spec = BoundedTensorSpec(
            (single_action_dim * plan_length + 2, ),
            minimum=np.concatenate((min_spec, np.array([1, 1]))),
            maximum=np.concatenate((max_spec,
                                    np.array([plan_length, plan_length]))))

        min_spec = np.tile(single_min, num_of_anchors)
        max_spec = np.tile(single_max, num_of_anchors)
        anchor_action_spec = BoundedTensorSpec(
            (single_action_dim * num_of_anchors, ),
            minimum=min_spec,
            maximum=max_spec)

        single_action_spec = BoundedTensorSpec((single_action_dim, ),
                                               minimum=single_min,
                                               maximum=single_max)

        self._rl_action_spec_anchor = (BoundedTensorSpec(
            shape=(), dtype='int64', maximum=1), anchor_action_spec)

        self._rl_action_spec_single = (BoundedTensorSpec(
            shape=(), dtype='int64', maximum=1), single_action_spec)

        self._action_spec = pad_action_spec
        self._single_action_spec = single_action_spec
        self._observation_spec = observation_spec
        self._gamma = gamma
        self._reward_clip_value = reward_clip_value

        self._num_of_anchors = num_of_anchors
        self._plan_length = plan_length
        self._mini_batch_length = mini_batch_length

        self._num_unroll_steps = num_unroll_steps
        self._target_fields = target_fields

        self._alg_render = alg_render

        self._action_to_world_conversion = action_to_world_conversion

        predict_state_spec = GenerativePlanningState(
            rl=rl.predict_state_spec,
            action=ext_action_spec,
            switch_flag=TensorSpec(shape=(), dtype='bool'),
            effective_plan_steps=TensorSpec(shape=(), dtype='int64'),
            full_plan_steps=TensorSpec(shape=(), dtype='int64'))

        rollout_state_spec = predict_state_spec._replace(
            rl=rl.rollout_state_spec,
            rl_discount=TensorSpec(()),
            rl_reward=TensorSpec(()),
            k=TensorSpec((), dtype='int64'),
            sample_rewards=TensorSpec(()))

        train_state_spec = GenerativePlanningState(rl=rl.train_state_spec)

        if repr_learner is not None:
            predict_state_spec = predict_state_spec._replace(
                repr=repr_learner.predict_state_spec)
            rollout_state_spec = rollout_state_spec._replace(
                repr=repr_learner.rollout_state_spec)
            train_state_spec = train_state_spec._replace(
                repr=repr_learner.train_state_spec)

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._repr_learner = repr_learner
        self._reward_normalizer = None
        if reward_normalizer_ctor is not None:
            self._reward_normalizer = reward_normalizer_ctor()
        self._rl = rl
        self._param_method = param_method

    def observe_for_replay(self, exp):
        # Do not observe data at every time step; customized observing
        pass

    def _should_switch_action(self, time_step: TimeStep, state):
        repeat_last_step = (state.effective_plan_steps == 0)
        flag = torch.as_tensor(
            repeat_last_step) | time_step.is_first() | state.switch_flag
        return flag

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        pose = self.get_pose(time_step.observation)
        prev_traj = self.get_prev_traj(time_step)

        @torch.no_grad()
        def _generate_new_action(time_step, state, pose):
            repr_state = ()
            rl_time_step = time_step

            observation, repr_state = rl_time_step.observation, ()
            if self._repr_learner is not None:
                repr_step = self._repr_learner.predict_step(
                    time_step, state.repr)
                observation = repr_step.output
                repr_state = repr_step.state

            # enter: world-to-ego
            prev_action, _ = self._action_to_ego(state.action, pose)
            prev_action = prev_action.permute(0, 2, 1).reshape(
                prev_action.shape[0], -1)

            rl_step = self._rl.predict_step(
                rl_time_step._replace(observation=observation),
                state.rl._replace(
                    plan=Plan(
                        full_plan=prev_action,
                        effective_plan_steps=state.effective_plan_steps)),
                epsilon_greedy, pose, prev_traj)
            rl_step = rl_step._replace(
                info=(rl_step.info, state.k, state.sample_rewards))

            # now the discrete output is switch flag
            switch_flag, action, q_values_prev_for_sel = rl_step.output

            # set steps to full, since we want to commit by default
            # using index representation [0, K-1], later move to [1, K]
            # shen setting values for state
            steps = torch.full_like(switch_flag, self._plan_length - 1)

            # convert to bool
            switch_flag = switch_flag.bool()

            action = self._action_to_world(action, pose)

            return GenerativePlanningState(
                action=action,
                switch_flag=switch_flag,
                effective_plan_steps=steps + 1,  # [0, K-1] -> [1, K]
                full_plan_steps=torch.full_like(steps, self._plan_length),  # K
                # k=torch.zeros_like(state.k),
                rl=rl_step.state,
                repr=repr_state), q_values_prev_for_sel

        # 1) run alg
        new_alg_state, q_values_prev_for_sel = _generate_new_action(
            time_step, state, pose)

        # 2) switch: update the state with switch_flag from alg, for state, we
        # still rely on its effective_plan_step etc to determie on the switch
        switch_state = state._replace(switch_flag=new_alg_state.switch_flag)
        switch_action = self._should_switch_action(time_step, switch_state)


        new_alg_state = new_alg_state._replace(switch_flag=switch_action)

        def _update_func(old_state, state):
            # avg plan commit steps
            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope(self._name):
                    alf.summary.histogram("plan_steps",
                                          old_state.effective_plan_steps)
                    alf.summary.scalar(
                        "plan_steps_avg",
                        old_state.effective_plan_steps.float().mean())

            return state

        # 3) use the switch condition to update
        new_state = conditional_update(
            target=switch_state,
            cond=switch_action,
            func=_update_func,
            old_state=state,
            state=new_alg_state)

        # 4) other processing such as plan step update
        effective_plan_steps = new_state.effective_plan_steps
        full_plan_steps = new_state.full_plan_steps

        new_state = new_state._replace(
            effective_plan_steps=effective_plan_steps - 1,
            full_plan_steps=full_plan_steps - 1)

        current_action = new_state.action

        current_action_ego, traj = self._action_to_ego(current_action, pose)

        steps_for_sel = (effective_plan_steps - 1 +
                         self._plan_length) % self._plan_length + 1

        action_spec = self._single_action_spec

        y_min = action_spec.minimum[0]
        y_max = action_spec.maximum[0]

        # world-to-ego case
        if self._alg_render:
            with alf.summary.scope("GPM"):
                img_size = 256
                # the actual action traj
                action_samples_img_actual = alf.summary.render.render_traj_set(
                    name="plan_actual",
                    data=current_action_ego,
                    effective_plan_steps=steps_for_sel,
                    x_range=[0, self._plan_length],
                    y_range=[y_min, y_max],
                    img_height=img_size,
                    img_width=img_size
                )

                step_img = alf.summary.render.render_bar(
                    name="switch", data=new_state.switch_flag)
            info_dict = dict(
                step_img=step_img,
                plan_old=self._rl._info["plan_old"],
                plan_new=self._rl._info["plan_new"],
                effective_steps_image=self._rl._info["effective_steps_image"],
                action_samples_img_actual=action_samples_img_actual,
            )
        else:
            info_dict = {}

        current_action_ego = current_action_ego.permute(
            0, 2, 1).reshape(*current_action.shape)

        # 2) preparing next step world plan by shifting
        shifted_traj = torch.cat(
            (traj[:, :, 1:], torch.zeros_like(traj[:, :, 0:1])), dim=2)

        new_state = new_state._replace(
            action=shifted_traj.permute(0, 2, 1).reshape(
                *current_action.shape))

        current_action_ego_concat = torch.cat(
            (current_action_ego, effective_plan_steps.float().unsqueeze(-1),
             full_plan_steps.float().unsqueeze(-1)),
            dim=-1)

        return AlgStep(
            output=current_action_ego_concat,
            state=new_state,
            info=info_dict)

    def add_zero_z(self, points):
        # points [B, d, L]
        padded_points = torch.cat((points, torch.zeros_like(points[:, 0:1])),
                                  dim=1)
        return padded_points

    def _action_to_world(self, action, pose):
        """Ego action to world
        Args:
            action: [B, L*d]
        Output:
            [B, L*d]
        """
        if self._action_to_world_conversion:
            # [B, L, d] -> [B, d, L]
            raw_action = action.reshape(action.shape[0], self._plan_length,
                                        -1).permute(0, 2, 1)
            world_plan = common.relative_to_world_position(
                pose, raw_action, keep_z_dim_same=True)
            action = world_plan.permute(0, 2, 1).reshape(action.shape[0], -1)
        return action

    def _action_to_ego(self, action, pose, reshape=True):
        """World action to ego
        Args:
            action: [B, L*d]
        Output:
            [B, d, L] # note that the shape is different from input
        """
        # also update plan execution
        # this should be done in the end
        if reshape:
            # [B, L*d]
            current_action_world = action

            # [B, L, d] -> [B, d, L]
            traj = current_action_world.reshape(current_action_world.shape[0],
                                                self._plan_length, -1).permute(
                                                    0, 2, 1)
        else:
            traj = action

        if self._action_to_world_conversion:
            # 1) convert world plan to ego as the current action
            current_action_ego = common.world_to_relative_position(
                pose, traj, keep_z_dim_same=True)
        else:
            current_action_ego = traj
        return current_action_ego, traj

    def _additional_proc_bf_output(self,
                                   current_action_ego=None,
                                   effective_plan_steps=None):
        # concat steps
        current_action_ego_concat = torch.cat(
            (current_action_ego, effective_plan_steps.float().unsqueeze(-1),
             full_plan_steps.float().unsqueeze(-1)),
            dim=-1)
        return current_action_ego_concat

    def rollout_step(self, time_step: TimeStep,
                     state: GenerativePlanningState):
        # has to run the algorithm first and then decide whether to switch

        pose = self.get_pose(time_step.observation)
        prev_traj = self.get_prev_traj(time_step)

        @torch.no_grad()
        def _generate_new_action(time_step, state, pose):
            rl_time_step = time_step

            observation, repr_state = rl_time_step.observation, ()
            if self._repr_learner is not None:
                repr_step = self._repr_learner.rollout_step(
                    time_step, state.repr)
                observation = repr_step.output
                repr_state = repr_step.state

            # enter: world-to-ego
            prev_action, _ = self._action_to_ego(state.action, pose)

            # [B, d, L] -> [B, L, d] -> [B, L,*d]
            prev_action = prev_action.permute(0, 2, 1).reshape(
                prev_action.shape[0], -1)

            rl_step = self._rl.rollout_step(
                rl_time_step._replace(observation=observation),
                state.rl._replace(
                    plan=Plan(
                        full_plan=prev_action,
                        effective_plan_steps=state.effective_plan_steps)))

            # unroll quantities for adjustment
            p_commit = rl_step.info.p_commit
            natural_switch = rl_step.info.natural_switch

            rl_step = rl_step._replace(
                info=(rl_step.info, state.k, state.sample_rewards))

            # now the discrete output is switch flag
            switch_flag, action, natural_switch = rl_step.output

            # set steps to full, since we want to commit by default
            # using index representation [0, K-1], later move to [1, K]
            # shen setting values for state
            steps = torch.full_like(switch_flag, self._plan_length - 1)

            # convert to bool
            switch_flag = switch_flag.bool()

            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope(self._name):
                    alf.summary.scalar("switch_flag",
                                       switch_flag.float().mean())

                    def mean_within_plan_value(natural_switch, switch_flag):
                        valid_mask = ~natural_switch
                        mean_val = (valid_mask * switch_flag.float()
                                    ).sum() / max(1.0,
                                                  valid_mask.float().sum())
                        return mean_val

                    if natural_switch.sum() > 0:
                        mean_switch = mean_within_plan_value(
                            natural_switch, switch_flag)
                        alf.summary.scalar("within_plan_switch", mean_switch)
                        alf.summary.scalar("within_plan_commit",
                                           1 - mean_switch)

            # exit: ego-to-world
            action = self._action_to_world(action, pose)

            return GenerativePlanningState(
                action=action,
                switch_flag=switch_flag,
                effective_plan_steps=steps + 1,  # [0, K-1] -> [1, K]
                full_plan_steps=torch.full_like(steps, self._plan_length),  # K
                k=torch.zeros_like(state.k),
                repr=repr_state,
                rl=rl_step.state,
                rl_reward=torch.zeros_like(state.rl_reward),
                sample_rewards=torch.zeros_like(state.sample_rewards),
                rl_discount=torch.ones_like(
                    state.rl_discount)), p_commit, natural_switch

        # 1) run alg
        new_alg_state, p_commit, natural_switch = _generate_new_action(
            time_step, state, pose)

        # 2) switch: update the state with switch_flag from alg, for state, we
        # still rely on its effective_plan_step etc to determie on the switch
        switch_state = state._replace(switch_flag=new_alg_state.switch_flag)
        switch_action = self._should_switch_action(time_step, switch_state)
        new_alg_state = new_alg_state._replace(switch_flag=switch_action)

        def _update_func(old_state, state):
            # avg plan commit steps
            if self._debug_summaries and alf.summary.should_record_summaries():
                with alf.summary.scope(self._name):
                    # old_state.effective_plan_steps is the steps remained
                    # need to do self._plan_length - old_state.effective_plan_steps
                    # to get the actual steps taken
                    steps_taken = self._plan_length - old_state.effective_plan_steps.float(
                    )
                    alf.summary.histogram("plan_steps", steps_taken)
                    alf.summary.scalar("plan_steps_avg", steps_taken.mean())

            return state

        # 3) use the switch condition to update
        new_state = conditional_update(
            target=switch_state,
            cond=switch_action,
            func=_update_func,
            old_state=state,
            state=new_alg_state)

        # 4) other processing such as plan step update
        effective_plan_steps = new_state.effective_plan_steps
        full_plan_steps = new_state.full_plan_steps

        new_state = new_state._replace(
            effective_plan_steps=effective_plan_steps - 1,
            full_plan_steps=full_plan_steps - 1)

        current_action = new_state.action

        current_action_ego, traj = self._action_to_ego(current_action, pose)

        # one step action
        onestep_current_action_ego = current_action_ego[..., 0]
        onestep_current_action_world = traj[..., 0]
        one_step_current_action = onestep_current_action_world

        # algo output for visualization and control
        current_action_ego = current_action_ego.permute(
            0, 2, 1).reshape(*current_action.shape)

        # 2) preparing next step world plan by shifting
        shifted_traj = torch.cat(
            (traj[:, :, 1:], torch.zeros_like(traj[:, :, 0:1])), dim=2)


        new_state = new_state._replace(
            action=shifted_traj.permute(0, 2, 1).reshape(
                *current_action.shape))

        # save one step action in world coordinate
        alg_step = AlgStep(
            output=(effective_plan_steps - 1, one_step_current_action),
            info=ReplanInfo(p_commit=p_commit, natural_switch=natural_switch))
        super(GenerativePlanningAgent, self).observe_for_replay(
            make_experience(
                time_step._replace(
                    observation=time_step.untransformed.observation), alg_step,
                state))

        # concat steps
        current_action_ego_concat = torch.cat(
            (current_action_ego, effective_plan_steps.float().unsqueeze(-1),
             full_plan_steps.float().unsqueeze(-1)),
            dim=-1)

        return AlgStep(output=current_action_ego_concat, state=new_state)

    def get_pose(self, obs):
        if self._action_to_world_conversion:
            pose = alf.nest.get_field(obs, 'observation.pose')
        else:
            pose = torch.zeros(obs.shape[0])
        return pose

    def get_prev_traj(self, exp):
        # prev traj in world coordinate
        try:
            prev_traj = alf.nest.get_field(exp.observation,
                                           'observation.prev_traj')
        except:
            prev_traj = torch.zeros(exp.reward.shape[0])
        return prev_traj

    def train_step(self, rl_exp: Experience, state: GenerativePlanningState):
        """Train the underlying RL algorithm ``self._rl``. Because in
        ``self.rollout_step()`` the replay buffer only stores info related to
        ``self._rl``, here we can directly call ``self._rl.train_step()``.

        Args:
            rl_exp (Experience): experiences that have been transformed to be
                learned by ``self._rl``.
            state (GenerativePlanningState):
        """
        # try:
        pose = self.get_pose(rl_exp.observation)
        prev_traj = self.get_prev_traj(rl_exp)

        repr_state = ()
        if self._repr_learner is not None:
            repr_step = self._repr_learner.train_step(rl_exp, state.repr)
            rl_exp = rl_exp._replace(observation=repr_step.output)
            repr_state = repr_step.state

            # process terminal observation
            with torch.no_grad():
                terminal_observation = rl_exp.rollout_info.terminal_observation
                term_exp = rl_exp._replace(observation=terminal_observation)
                term_transformed_exp = self.transform_experience(term_exp)

                pre_processed_exp = self._rl.preprocess_experience(
                    term_transformed_exp)
                repr_step = self._repr_learner.rollout_step(
                    pre_processed_exp, ())
                terminal_observation = repr_step.output
                rl_exp = rl_exp._replace(
                    rollout_info=rl_exp.rollout_info._replace(
                        terminal_observation=terminal_observation))

        rl_step = self._rl.train_step(rl_exp, state.rl, pose, prev_traj)

        new_state = GenerativePlanningState(rl=rl_step.state, repr=repr_state)
        rl_info = rl_step.info

        new_info = rl_info

        return rl_step._replace(state=new_state, info=new_info)

    def batch_to_temporal(self, batch):
        # batch: [T*B, ...]
        # [T, B, ...]
        tc_batch = batch.reshape(self._mini_batch_length, -1, *batch.shape[1:])
        return tc_batch

    def calc_loss(self, rl_experience, rl_info):
        """Calculate the loss for training ``self._rl``."""
        return self._rl.calc_loss(rl_experience, rl_info)

    def after_update(self, rl_exp, rl_info):
        """Call ``self._rl.after_update()``."""
        self._rl.after_update(rl_exp, rl_info)

    def summarize_rollout(self, experience):
        """Generate summaries for rollout.

        Args:
            experience (Experience): experience collected from ``rollout_step()``.
        """

        if self._debug_summaries:
            action_for_summary = experience.action[..., :-2]

            cnt_action_spec = self._rl_action_spec[-1]
            actions = action_for_summary.reshape(
                -1, self._plan_length, cnt_action_spec.shape[0]).permute(
                    0, 2, 1)

            action_t = actions[..., 0]
            summary_utils.summarize_action(action_t, cnt_action_spec,
                                           "rollout_action")
            self.summarize_reward("rollout_reward/extrinsic",
                                  experience.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(experience.rollout_info,
                                        'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(
                    action_distributions=field[0], name="rollout_action_dist")

    def summarize_train(self, experience, train_info, loss_info, params):
        """Overwrite the function because the training action spec is
        different from the rollout action spec.
        """
        Algorithm.summarize_train(self, experience, train_info, loss_info,
                                  params)

        if self._debug_summaries:
            self.summarize_reward("training_reward", experience.reward)

        if self._config.summarize_action_distributions:
            field = alf.nest.find_field(train_info, 'action_distribution')
            if len(field) == 1:
                summary_utils.summarize_action_dist(field[0])

    @torch.no_grad()
    def preprocess_experience(self, experience: Experience):
        """Fill experience.rollout_info with PredictiveRepresentationLearnerInfo

        Note that the shape of experience is [B, T, ...].

        The target is a Tensor (or a nest of Tensors) when there is only one
        decoder. When there are multiple decorders, the target is a list,
        and each of its element is a Tensor (or a nest of Tensors), which is
        used as the target for the corresponding decoder.

        Note:
            need to get sequence of action, reward and terminal state.
            Also different from predictive learning, we retrieve sequence
            of length num_unroll_steps instead of num_unroll_steps + 1.

            Need to enable
            ReplayBuffer.keep_episodic_info=True

        """

        assert experience.batch_info != ()
        batch_info: BatchInfo = experience.batch_info
        replay_buffer: ReplayBuffer = experience.replay_buffer
        batch_size = experience.step_type.shape[0]
        mini_batch_length = experience.step_type.shape[1]

        with alf.device(replay_buffer.device):
            # [B, 1]
            positions = convert_device(batch_info.positions).unsqueeze(-1)
            # [B, 1]
            env_ids = convert_device(batch_info.env_ids).unsqueeze(-1)

            # get action for the first time
            # [B, 1, 1]
            env_ids = env_ids.unsqueeze(-1)
            positions = positions.unsqueeze(-1)

            # [B, 1]
            steps_to_episode_end = replay_buffer.steps_to_episode_end(
                positions, env_ids)

            # [B, T]
            episode_end_positions = positions + steps_to_episode_end

            valid_positions = torch.min(positions, episode_end_positions)
            new_observation = replay_buffer.get_field('observation', env_ids,
                                                      valid_positions)

            # [B, T, unroll_steps + 1]
            # note that the position starts from the current step, including
            # the current step
            positions_full = positions + torch.arange(
                0, self._num_unroll_steps + 1)

            # since we are focusing on the current time step and
            # trying to extract valid temporal sequences >=2,
            # here the valid mask should be <. NOT <=.
            # Otherwise, using the last step as the current step will
            # incur losses
            mask = positions_full < episode_end_positions
            valid_mask = positions_full <= episode_end_positions

            positions_full_clipped = torch.min(positions_full,
                                               episode_end_positions)

            # [B, T, unroll_steps + 1] --> [B, T, unroll_steps]
            # starting from the current step actions
            action = replay_buffer.get_field('action', env_ids,
                                             positions_full_clipped[..., 0:-1])

            # convert plans from world coordinate to current ego-centric coordinate
            pose = self.get_pose(new_observation)

            # action[1]: [B, T=1, L, d]
            cnt_action = action[1]
            cnt_action = cnt_action.reshape(-1, cnt_action.shape[2],
                                            cnt_action.shape[3]).permute(
                                                0, 2, 1)
            pose = pose.reshape(-1, pose.shape[-1])

            # [B, d, L]
            current_action_ego, traj = self._action_to_ego(
                cnt_action, pose, reshape=False)

            current_action_ego = current_action_ego.permute(
                0, 2, 1).reshape(*action[1].shape)

            # new action
            action = (action[0], current_action_ego)

            # [B, T, 1]: +1: ind -> step
            rand_num = np.random.rand()
            if rand_num < 0:
                sampled_unroll_step = action[0][..., 0] + 1
            elif rand_num < 0:
                sampled_unroll_step = torch.full((batch_size, 1),
                                                 self._num_unroll_steps,
                                                 dtype=action[0][..., 0].dtype)
            elif rand_num < 0:
                sampled_unroll_step = torch.full((batch_size, 1),
                                                 1,
                                                 dtype=action[0][..., 0].dtype)
            elif rand_num < 0:
                sampled_unroll_step = torch.full((batch_size, 1),
                                                 self._plan_length,
                                                 dtype=action[0][..., 0].dtype)
            else:
                h_batch_size = batch_size // 2
                # maintain half of the batch as one-step plans
                # to ensure the value learning quality
                sampled_unroll_step0 = torch.randint(
                    low=1, high=2, size=(batch_size - h_batch_size, 1))

                sampled_unroll_step = torch.randint(
                    low=1,
                    high=self._num_unroll_steps + 1,
                    size=(h_batch_size, 1))

                sampled_unroll_step = torch.cat(
                    [sampled_unroll_step0, sampled_unroll_step], dim=0)

                mask = positions_full < episode_end_positions
                valid_mask = positions_full <= episode_end_positions

            unroll_end_position = positions + sampled_unroll_step.unsqueeze(1)

            unroll_end_position = torch.min(unroll_end_position,
                                            episode_end_positions)



            # [B, T, unroll_steps + 1]
            # relative displacement wrt first position
            position_displacement = (positions_full - positions_full[..., 0:1])

            valid_non_last_mask = positions_full < unroll_end_position

            last_step_mask = positions_full == unroll_end_position

            # [B, T, unroll_steps + 1]
            reward = replay_buffer.get_field('reward', env_ids,
                                             positions_full_clipped)

            # get terminal state/observation
            # [B, T]
            terminal_observation = replay_buffer.get_field(
                'observation', env_ids, unroll_end_position)

            terminal_observation = nest.map_structure(
                lambda tensor: tensor.squeeze(2), terminal_observation)

            # get terminal discount
            # [B, T]
            terminal_discount = replay_buffer.get_field(
                'discount', env_ids, unroll_end_position)

            discount = valid_non_last_mask + last_step_mask * terminal_discount

            step_type = replay_buffer.get_field('step_type', env_ids,
                                                positions_full_clipped)

            rollout_info = GenerativePlanningInfo(
                action=action,
                reward=reward,
                mask=mask,
                valid_mask=valid_mask,
                last_step_mask=last_step_mask,
                terminal_observation=terminal_observation,
                discount=discount,
                sampled_unroll_step=sampled_unroll_step,
                p_commit=experience.rollout_info.p_commit,
                natural_switch=experience.rollout_info.natural_switch)

        rollout_info = convert_device(rollout_info)

        if self._reward_normalizer:
            reward0 = rollout_info.reward
            reward_vec = reward0.permute(0, 2, 1).reshape(
                reward0.shape[0] * reward0.shape[2], -1)
            reward_vec_valid = reward_vec[valid_mask.view(-1)]

            self._reward_normalizer.update(reward_vec_valid)

            reward = self._reward_normalizer.normalize(
                reward_vec, clip_value=self._reward_clip_value)
            reward = reward.reshape(reward0.shape[0], reward0.shape[2],
                                    -1).permute(0, 2, 1)

            rollout_info = rollout_info._replace(reward=reward)


        new_obs = nest.map_structure(
            lambda tensor: convert_device(tensor.squeeze(1)), new_observation)
        rl_exp = experience._replace(
            rollout_info=rollout_info,
            observation=new_obs
        )

        # make sure no old fields been transformed multiple times
        rl_exp = self.transform_experience(rl_exp)
        return self._rl.preprocess_experience(rl_exp)
