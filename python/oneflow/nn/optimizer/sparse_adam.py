# """
# Copyright 2020 The OneFlow Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """
# import collections
# from typing import Callable, Dict, Iterator, List, Tuple, Union

# import oneflow as flow
# from oneflow.nn.optimizer.optimizer import Optimizer, ParamGroup
# from oneflow.nn.parameter import Parameter

# """
# class LazyAdam(Optimizer):
#     """
#     The optimizer of the LazyAdam algorithm.

#     This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates and the 2nd-moment estimates of the gradient.

#     The difference between Adam optimizer and LazyAdam optimizer is that LazyAdam only updates the element that has gradient in the current batch, it is faster than Adam optimizer.

#     .. math::

#         & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

#         & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

#         & \\hat{g} = learning\\_rate*\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}

#         & param_{new} = param_{old} - \\hat{g}

#     Args:
#         lr_scheduler (LrScheduler): The scheduler of learning rate.
#         beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\\beta_1`). Defaults to 0.9.
#         beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\\beta_2`). Defaults to 0.999.
#         epsilon ([type], optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-8.
#         loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
#         grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
#         train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
#         loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
#         variables(Optional[
#             Union[Sequence[Text], Callable[[], Sequence[Text]]]
#         ]): maintained variables.

#     For example:

#     .. code-block:: python

#         import oneflow.compatible.single_client as flow
#         import oneflow.compatible.single_client.typing as tp

#         @flow.global_function(type="train")
#         def train_job(
#             images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
#             labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
#         ) -> tp.Numpy:
#             with flow.scope.placement("gpu", "0:0"):
#                 logits = lenet(images, train=True)
#                 loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
#                     labels, logits, name="softmax_loss"
#                 )
#             # Set learning rate as 0.001
#             lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
#             # Set LazyAdam optimizer
#             flow.optimizer.LazyAdam(lr_scheduler).minimize(loss)

#             return loss

#     """

#     def __init__(
#         self,
#         lr_scheduler: LrScheduler,
#         beta1: float = 0.9,
#         beta2: float = 0.999,
#         epsilon: float = 1e-08,
#         loss_scale_factor: Optional[float] = None,
#         grad_clipping: Optional[ClipGradientConf] = None,
#         train_step_lbn: Optional[Text] = None,
#         loss_scale_policy: Optional[LossScalePolicy] = None,
#         variables: Optional[
#             Union[Sequence[Text], Callable[[], Sequence[Text]]]
#         ] = GetVariablesForCurrentJob,
#     ):
#         super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
#         self.lr_scheduler = lr_scheduler
#         self.grad_clipping = grad_clipping
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.variables = variables

#     def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
#         optimizer_conf = train_conf.mutable_optimizer_conf().Add()
#         self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
#         if self.grad_clipping is not None:
#             optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
#         optimizer_conf.mutable_lazy_adam_conf().set_beta1(self.beta1)
#         optimizer_conf.mutable_lazy_adam_conf().set_beta2(self.beta2)
#         optimizer_conf.mutable_lazy_adam_conf().set_epsilon(self.epsilon)
#         for variable in self.Variables():
#             optimizer_conf.add_variable_op_names(variable)

# """

# class SparseAdam(Optimizer):
#     def __init__(
#         self,
#         parameters: Union[Iterator[Parameter], List[Dict]],
#         lr: float = 0.001,
#         betas: Tuple[float, float] = (0.9, 0.999),
#         eps: float = 1e-08,
#     ):
#         super().__init__()
#         assert lr >= 0.0, f"Invalid learning rate: {lr}"
#         assert eps >= 0.0, f"Invalid epsilon value: {eps}"
#         assert (
#             betas[0] >= 0.0 and betas[0] < 1.0
#         ), f"Invalid beta parameter at index 0: {betas[0]}"
#         assert (
#             betas[1] >= 0.0 and betas[1] < 1.0
#         ), f"Invalid beta parameter at index 1: {betas[1]}"
#         self._default_options["lr"] = lr
#         self._default_options["eps"] = eps
#         self._default_options["betas"] = betas
#         if isinstance(parameters, collections.abc.Iterator):
#             self.param_groups.append(ParamGroup(parameters, self._default_options))
#         else:
#             for param in parameters:
#                 self.param_groups.append(ParamGroup(param, self._default_options))
#         for param_group in self.param_groups:
#             for param in param_group.parameters:
#                 assert param.is_leaf, "parameters must be leaf tensor"
#                 self._state[param] = dict()
#                 self._state[param]["exp_avg"] = flow.zeros_like(param)
#                 self._state[param]["exp_avg_sq"] = flow.zeros_like(param)
#         self._op = (
#             flow.builtin_op("indexed_slices_adam_update")
#             .Input("model")
#             .Input("model_diff_indices")
#             .Input("model_diff")
#             .Input("m")
#             .Input("v")
#             .Attr("weight_decay", 0.0)
#             .Build()
#         )
#         """
#         REGISTER_NO_GRAD_USER_OP("indexed_slices_adam_update")
#         .Input("model")
#         .Input("model_diff_indices")
#         .Input("model_diff_values")
#         .Input("learning_rate")
#         .Input("m")
#         .Input("v")
#         .Attr<float>("beta1", 0.9)
#         .Attr<float>("beta2", 0.999)
#         .Attr<float>("epsilon", 1e-8)
#         .Attr<float>("weight_decay", 0.0)
#         """

#     def step(self, closure: Callable = None):
#         """Performs a single optimization step.

#         Args:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         with flow.no_grad():
#             loss = None
#             if closure is not None:
#                 loss = closure()
#             for param_group in self.param_groups:
#                 kwargs = {
#                     "learning_rate_val": param_group["lr"],
#                     "beta1": param_group["betas"][0],
#                     "beta2": param_group["betas"][1],
#                     "epsilon": param_group["eps"],
#                 }
#                 for param in param_group.parameters:
#                     if param.grad is None:
#                         continue
#                     m_tensor = self._state[param]["exp_avg"]
#                     v_tensor = self._state[param]["exp_avg_sq"]
#                     self._op(param, param.grad, m_tensor, v_tensor, **kwargs)
#             self._state["step"] = self._state["step"] + 1
#             return loss