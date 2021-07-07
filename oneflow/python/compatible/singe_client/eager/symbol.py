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

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.core.job.placement_pb2 as placement_pb
import functools


class Symbol(object):
    def __init__(self, symbol_id, data):
        self.symbol_id_ = symbol_id
        self.data_ = data

    @property
    def symbol_id(self):
        return self.symbol_id_

    @property
    def data(self):
        return self.data_
