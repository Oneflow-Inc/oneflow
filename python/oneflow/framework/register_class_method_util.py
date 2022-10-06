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
import oneflow._oneflow_internal
import oneflow.framework.check_point_v2 as check_point_v2
import oneflow.framework.generator as generator
import oneflow.framework.tensor as tensor_util


def RegisterMethod4Class():
    tensor_util.RegisterMethods()
    check_point_v2.RegisterMethods()
