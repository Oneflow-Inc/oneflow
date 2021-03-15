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
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("nn.init.uniform_")
def uniform_(tensor, a=0.0, b=1.0):
    # TODO(jianhao): add with torch.no_grad() when autograd is ready
    tensor.uniform_(a, b)


@oneflow_export("nn.init.normal_")
def normal_(tensor, mean=0.0, std=1.0):
    tensor.normal_(mean, std)


@oneflow_export("nn.init.constant_")
def constant_(tensor, val):
    tensor.fill_(val)


@oneflow_export("nn.init.ones_")
def ones_(tensor):
    tensor.fill_(1)


@oneflow_export("nn.init.zeros_")
def zeros_(tensor):
    tensor.fill_(0)
