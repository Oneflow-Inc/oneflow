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

import oneflow.core.common.data_type_pb2 as data_type_conf_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("truncated_normal")
def truncated_normal_initializer(
    stddev: float = 1.0,
) -> initializer_conf_util.InitializerConf:
    initializer = initializer_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "std", float(stddev))

    return initializer
