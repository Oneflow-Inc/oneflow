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

import numpy as np


class Blob(object):
    def __init__(self, ndarray=None):
        self.ndarray_ = ndarray

    def ndarray(self):
        return self.ndarray_

    def set_ndarray(self, ndarray):
        self.ndarray_ = ndarray

    def __getattr__(self, attr):
        return getattr(self.ndarray_, attr)


no_override_field = set(
    [
        "__class__",
        "__doc__",
        "__new__",
        "__init__",
        "__del__",
        "__call__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__dir__",
        "__get__",
        "__set__",
        "__delete__",
    ]
)


def MakeBlobMethod(field_name):
    def ConvertOtherArgs(args):
        return [x.ndarray_ if isinstance(x, Blob) else x for x in args]

    return lambda self, *args: getattr(self.ndarray_, field_name)(
        *ConvertOtherArgs(args)
    )


for field_name in dir(np.ndarray):
    if field_name.startswith("__") == False:
        continue
    if field_name in no_override_field:
        continue
    setattr(Blob, field_name, MakeBlobMethod(field_name))
