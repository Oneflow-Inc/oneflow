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
import oneflow as flow
from oneflow.nn.optimizer.adam import Adam


class SmartDecayAdam(Adam):
    """Implements SmartDecayAdam algorithm.
       The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
       For Sparse Embedding Table in OneEmbedding, implement the SmartDecayAdam algorithm.
       For other models, it is same as Adam.
    """

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = super()._generate_conf_for_graph(train_conf, vars_conf)
        for opt_conf in new_opt_confs:
            opt_conf.adam_conf.smart_decay = True
