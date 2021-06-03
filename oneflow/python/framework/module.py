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
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.id_util as id_util


@oneflow_export("deprecated.nn.Module")
class Module(object):
    def __init__(self, name=None):
        if name is None:
            name = id_util.UniqueStr("Module_")
        self.module_name_ = name
        self.call_seq_no_ = 0

    @property
    def module_name(self):
        return self.module_name_

    @property
    def call_seq_no(self):
        return self.call_seq_no_

    # only for overriding
    # do not call module.foward(*args) directly
    def forward(self, *args):
        raise NotImplementedError()

    def __call__(self, *args):
        ret = self.forward(*args)
        self.call_seq_no_ = self.call_seq_no_ + 1
        return ret

    def __del__(self):
        assert (
            getattr(type(self), "__call__") is Module.__call__
        ), "do not override __call__"
