"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
import math
from typing import Callable, Dict, Iterator, List, Tuple, Union

import oneflow as flow
from oneflow.optim.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter


class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates and the 2nd-moment estimates of gradient.

    the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}

        & param_{new} = param_{old} - \\hat{g}

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm. (default: False) 
        do_bias_correction (bool, optional): whether to do bias correction (default: True)
        contiguous_params (bool, optional): whether to use contiguous ParamGroup 
            which puts all parameters of the same type, device and group into the
            same tensor and update them together. (default: False)
        fused (bool, optional): whether to divide all the parameters into several groups, then
            update each group of parameters with the fused kernel. (default: False)

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    For example: 

    Example 1: 

    .. code-block:: python 

        # Assume net is a custom model. 
        adam = flow.optim.Adam(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            adam.step()
            adam.zero_grad()

    Example 2: 

    .. code-block:: python 

        # Assume net is a custom model. 
        adam = flow.optim.Adam(
            [
                {
                    "params": net.parameters(),
                    "lr": learning_rate,
                    "clip_grad_max_norm": 0.5,
                    "clip_grad_norm_type": 2.0,
                }
            ],
        )

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            adam.clip_grad()
            adam.step()
            adam.zero_grad()

    If you want to use clip_grad, you can refer this example. 

    For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`. 

    
    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        amsgrad: bool = False,
        do_bias_correction: bool = True,
        contiguous_params: bool = False,
        fused: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert (
            betas[0] >= 0.0 and betas[0] < 1.0
        ), f"Invalid beta parameter at index 0: {betas[0]}"
        assert (
            betas[1] >= 0.0 and betas[1] < 1.0
        ), f"Invalid beta parameter at index 1: {betas[1]}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        options = dict()
        options["lr"] = lr
        options["eps"] = eps
        options["betas"] = betas
        options["weight_decay"] = weight_decay
        options["amsgrad"] = amsgrad
        options["bias_correction1"] = 1.0
        options["bias_correction2"] = 1.0
        options["do_bias_correction"] = do_bias_correction
        options["contiguous_params"] = contiguous_params
        options["fused"] = fused
        super().__init__(params, options)

        for param_group in self.param_groups:
            if param_group["contiguous_params"]:
                param_list = param_group.contiguous_parameters
            else:
                param_list = param_group.parameters

            for param in param_list:
                assert param.is_leaf, "parameters must be leaf tensor"
                self.state[param] = dict()

                if param_group["fused"] and param_group["amsgrad"]:
                    warnings.warn("Fused Adam is not supported when amsgrad=True.")
                    param_group["fused"] = False

                if param_group["fused"] and not param.is_cuda:
                    warnings.warn("Fused Adam only support cuda parameters.")
                    param_group["fused"] = False

        self._op_with_amsgrad = (
            flow.stateful_op("adam_update")
            .Input("model")
            .Input("model_diff")
            .Input("m")
            .Input("v")
            .Input("max_v")
            .Build()
        )

        self._op_without_amsgrad = (
            flow.stateful_op("adam_update")
            .Input("model")
            .Input("model_diff")
            .Input("m")
            .Input("v")
            .Build()
        )

    def _single_tensor_update(self, param_group):
        kwargs = {
            "learning_rate": param_group["lr"],
            "bias_correction1": param_group["bias_correction1"],
            "bias_correction2": param_group["bias_correction2"],
            "l2": param_group["weight_decay"],
            "beta1": param_group["betas"][0],
            "beta2": param_group["betas"][1],
            "epsilon": param_group["eps"],
            "do_bias_correction": param_group["do_bias_correction"],
            "amsgrad": param_group["amsgrad"],
        }

        if param_group["contiguous_params"]:
            param_list = param_group.contiguous_parameters
        else:
            param_list = param_group.parameters

        for param in param_list:
            if param.grad is None:
                continue
            if "exp_avg" not in self.state[param]:
                self.state[param]["exp_avg"] = flow.zeros_like(param)
            if "exp_avg_sq" not in self.state[param]:
                self.state[param]["exp_avg_sq"] = flow.zeros_like(param)
            if param_group["amsgrad"]:
                if "max_exp_avg_sq" not in self.state[param]:
                    self.state[param]["max_exp_avg_sq"] = flow.zeros_like(param)

            m_tensor = self.state[param]["exp_avg"]
            v_tensor = self.state[param]["exp_avg_sq"]

            if param_group["amsgrad"]:
                max_v_tensor = self.state[param]["max_exp_avg_sq"]
                flow._C.dispatch_adam_update(
                    self._op_with_amsgrad,
                    (param, param.grad, m_tensor, v_tensor, max_v_tensor),
                    **kwargs,
                )
            else:
                flow._C.dispatch_adam_update(
                    self._op_without_amsgrad,
                    (param, param.grad, m_tensor, v_tensor),
                    **kwargs,
                )

    def _fused_update(self, param_group):
        param_list = []
        param_grad_list = []
        m_tensor_list = []
        v_tensor_list = []

        for param in param_group.parameters:
            if param.grad is None:
                continue

            if "exp_avg" not in self.state[param]:
                self.state[param]["exp_avg"] = flow.zeros_like(param)
            if "exp_avg_sq" not in self.state[param]:
                self.state[param]["exp_avg_sq"] = flow.zeros_like(param)
            if param_group["amsgrad"]:
                if "max_exp_avg_sq" not in self.state[param]:
                    self.state[param]["max_exp_avg_sq"] = flow.zeros_like(param)

            param_list.append(param)
            param_grad_list.append(param.grad)
            m_tensor_list.append(self.state[param]["exp_avg"])
            v_tensor_list.append(self.state[param]["exp_avg_sq"])

        flow._C.multi_tensor_adam_update(
            model=param_list,
            model_diff=param_grad_list,
            m=m_tensor_list,
            v=v_tensor_list,
            learning_rate_val=param_group["lr"],
            l2=param_group["weight_decay"],
            beta1=param_group["betas"][0],
            beta2=param_group["betas"][1],
            bias_correction1_val=param_group["bias_correction1"],
            bias_correction2_val=param_group["bias_correction2"],
            do_bias_correction=param_group["do_bias_correction"],
            scale=1.0,
            weight_decay=0.0,
            epsilon=param_group["eps"],
        )

    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with flow.no_grad():
            loss = None
            if closure is not None:
                with flow.enable_grad():
                    loss = closure()

            for param_group in self.param_groups:
                if param_group["do_bias_correction"]:
                    param_group["bias_correction1"] = 1.0 - math.pow(
                        param_group["betas"][0], self.state["step"] + 1
                    )
                    param_group["bias_correction2"] = 1.0 - math.pow(
                        param_group["betas"][1], self.state["step"] + 1
                    )

                if param_group["fused"]:
                    self._fused_update(param_group)
                else:
                    self._single_tensor_update(param_group)

            self.state["step"] += 1

            return loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = []
        for param_group in self.param_groups:
            assert (
                param_group["contiguous_params"] != True
            ), "contiguous_params cannot be used in graph"

            optimizer_conf = train_conf.optimizer_conf.add()

            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )
            l2 = param_group["weight_decay"]
            beta1 = param_group["betas"][0]
            beta2 = param_group["betas"][1]

            epsilon = param_group["eps"]
            do_bias_correction = param_group["do_bias_correction"]
            amsgrad = param_group["amsgrad"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.adam_conf.beta1 = beta1
            optimizer_conf.adam_conf.beta2 = beta2
            optimizer_conf.adam_conf.epsilon = epsilon
            optimizer_conf.adam_conf.do_bias_correction = do_bias_correction
            optimizer_conf.adam_conf.amsgrad = amsgrad

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs

    @property
    def support_sparse(self):
        return True
