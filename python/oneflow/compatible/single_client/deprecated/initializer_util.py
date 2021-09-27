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
from oneflow.core.common import data_type_pb2 as data_type_conf_util
from oneflow.core.job import initializer_conf_pb2 as initializer_conf_util
from oneflow.core.operator import op_conf_pb2 as op_conf_util


def truncated_normal_initializer(
    stddev: float = 1.0,
) -> initializer_conf_util.InitializerConf:
    initializer = initializer_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "std", float(stddev))
    return initializer
