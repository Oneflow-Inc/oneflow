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


class GradScaler(object):
    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    ):
        self._init_scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        if self._backoff_factor != 1.0 / self._growth_factor:
            raise ValueError(
                "Only support 1.0/growth_factor as backoff_factor at the moment, "
                "got {}".format(backoff_factor)
            )
        self._growth_interval = growth_interval

    def _generate_conf_for_graph(self, train_conf):
        train_conf.dynamic_loss_scale_policy.initial_loss_scale = self._init_scale
        train_conf.dynamic_loss_scale_policy.increment_period = self._growth_interval
        train_conf.dynamic_loss_scale_policy.multiplier = self._growth_factor


class StaticGradScaler(object):
    def __init__(self, scale_factor):
        if scale_factor <= 0.0:
            raise ValueError("StaticGradScaler's scale_factor must > 0.0")

        self._scale_factor = scale_factor

    def _generate_conf_for_graph(self, train_conf):
        train_conf.loss_scale_factor = self._scale_factor
