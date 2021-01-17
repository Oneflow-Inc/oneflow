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

import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow.python.eager.symbol as symbol_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow_api


class OpKernelObject(oneflow_api.Object):
    def __init__(self, object_id, op_conf, release):
        oneflow_api.Object.__init__(self, object_id, _GetOpParallelSymbol(op_conf))
        self.op_conf_ = op_conf
        self.scope_symbol_ = _GetScopeSymbol(op_conf)
        self.release_ = []
        if release is not None:
            self.release_.append(release)

    @property
    def op_conf(self):
        return self.op_conf_

    @property
    def scope_symbol(self):
        return self.scope_symbol_

    def __del__(self):
        for release in self.release_:
            release(self)
        self.release_ = []


def _GetScopeSymbol(op_conf):
    assert op_conf.HasField("scope_symbol_id")
    return oneflow_api.GetScopeSymbol(op_conf.scope_symbol_id)


def _GetOpParallelSymbol(op_conf):
    assert op_conf.HasField("scope_symbol_id")
    symbol_id = c_api_util.GetOpParallelSymbolId(op_conf)
    return oneflow_api.GetPlacementSymbol(symbol_id)
