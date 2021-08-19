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


class LossScalePolicy(objcet):
    def generate_conf_for_graph(self, train_conf):
        raise NotImplementedError()


class StaticLossScalePolicy(LossScalePolicy):
    def __init__(self, loss_scale_factor: float):
        super().__init__()
        self.loss_scale_factor = loss_scale_factor

    def generate_conf_for_graph(self, train_conf):
        train_conf.loss_scale_factor = self.loss_scale_factor


class DynamicLossScalePolicy(LossScalePolicy):
    def __init__(
        self, initial_loss_scale=2 ** 30, increment_period=2000, multiplier=2.0
    ):
        super().__init__()
        self.initial_loss_scale = initial_loss_scale
        self.increment_period = increment_period
        self.multiplier = multiplier

    def generate_conf_for_graph(self, train_conf):
        train_conf.mutable_dynamic_loss_scale_policy().set_initial_loss_scale(
            self.initial_loss_scale
        )
        train_conf.mutable_dynamic_loss_scale_policy().set_increment_period(
            self.increment_period
        )
        train_conf.mutable_dynamic_loss_scale_policy().set_multiplier(self.multiplier)
