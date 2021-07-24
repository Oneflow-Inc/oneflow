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
from typing import Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Union[str, flow.device] = None,
        dtype: flow.dtype = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        if self.affine:
            self.weight = flow.nn.Parameter(
                flow.Tensor(num_features, device=self.device)
            )
            self.bias = flow.nn.Parameter(flow.Tensor(num_features, device=self.device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", flow.Tensor(num_features, device=self.device)
            )
            self.register_buffer(
                "running_var", flow.Tensor(num_features, device=self.device)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.fill_(0)
            self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            flow.nn.init.ones_(self.weight)
            flow.nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )

    def forward(self, x):
        if self.dtype is None:
            self.dtype = x.dtype
        if self.device is None:
            self.device = x.device
        self._check_input_dim(x)
        reduce_axis = []
        for dim in range(len(x.shape)):
            if dim != 1:
                reduce_axis.append(dim)
        mean = x.mean(dim=reduce_axis, keepdim=False)
        variance = x.var(dim=reduce_axis, keepdim=False)
        if x.device == flow.device("cpu"):
            if self.training and self.track_running_stats:
                running_mean = (
                    self.momentum * self.running_mean + (1 - self.momentum) * mean
                )
                running_var = (
                    self.momentum * self.running_var + (1 - self.momentum) * variance
                )
                self.__setattr__("running_mean", flow.Tensor(running_mean))
                self.__setattr__("running_var", flow.Tensor(running_var))
            else:
                mean = mean if self.running_mean is None else self.running_mean
                variance = variance if self.running_var is None else self.running_var
            axis = 1
            params_shape = [x.shape[axis]]
            weight = self.weight
            bias = self.bias
            if len(mean.shape) == 1:
                nd_params_shape = [1] * len(x.shape)
                nd_params_shape[axis] = params_shape[0]
                mean = mean.reshape(shape=nd_params_shape)
                variance = variance.reshape(shape=nd_params_shape)
                if self.weight and params_shape[0] == self.weight.nelement():
                    weight = self.weight.reshape(shape=nd_params_shape)
                if self.bias and params_shape[0] == self.bias.nelement():
                    bias = self.bias.reshape(shape=nd_params_shape)
            elif len(mean.shape) == len(x.shape):
                pass
            else:
                raise ValueError(
                    "shape of mean and variance should be 1D or has number of axes and x's"
                )
            variance += self.eps
            normalized = (x - mean) * variance.rsqrt()
            affined = normalized
            if self.weight:
                affined = affined * weight
            if self.bias:
                affined = affined + bias
            return affined.to(dtype=self.dtype)
        else:
            res = flow.F.normalization(
                x,
                self.running_mean if self.track_running_stats else mean,
                self.running_var if self.track_running_stats else variance,
                self.weight,
                self.bias,
                axis=1,
                epsilon=self.eps,
                momentum=self.momentum,
                is_training=self.training,
            )
            return res.to(dtype=self.dtype, device=self.device)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
