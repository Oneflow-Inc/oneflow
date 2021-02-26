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
import oneflow_api
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.device as oneflow_device


oneflow_export("LocalTensor")(oneflow_api.LocalTensor)

oneflow_export("ConsistentTensor")(oneflow_api.ConsistentTensor)

