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
from __future__ import absolute_import
import oneflow.compatible.single_client as flow
import oneflow._oneflow_internal
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("Generator")
def create_generator(device=None):
    if device is None:
        device = "auto"
    return oneflow._oneflow_internal.create_generator(device)


@oneflow_export("default_generator")
def default_generator(device=None):
    if device is None:
        device = "auto"
    return oneflow._oneflow_internal.default_generator(device)


@oneflow_export("manual_seed")
def manual_seed(seed):
    oneflow._oneflow_internal.manual_seed(seed)
