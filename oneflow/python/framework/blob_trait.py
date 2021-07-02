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

import oneflow


def __add__(self, rhs):
    return oneflow.math.add(self, rhs)


def __radd__(self, lhs):
    return oneflow.math.add(lhs, self)


def __sub__(self, rhs):
    return oneflow.math.subtract(self, rhs)


def __rsub__(self, lhs):
    return oneflow.math.subtract(lhs, self)


def __mul__(self, rhs):
    return oneflow.math.multiply(self, rhs)


def __rmul__(self, lhs):
    return oneflow.math.multiply(lhs, self)


def __truediv__(self, rhs):
    return oneflow.math.divide(self, rhs)


def __rtruediv__(self, lhs):
    return oneflow.math.divide(lhs, self)


def __div__(self, rhs):
    return oneflow.math.divide(self, rhs)


def __mod__(self, rhs):
    return oneflow.math.mod(self, rhs)


def __eq__(self, rhs):
    return oneflow.math.equal(self, rhs)


def __ne__(self, rhs):
    return oneflow.math.not_equal(self, rhs)


def __lt__(self, rhs):
    return oneflow.math.less(self, rhs)


def __le__(self, rhs):
    return oneflow.math.less_equal(self, rhs)


def __gt__(self, rhs):
    return oneflow.math.greater(self, rhs)


def __ge__(self, rhs):
    return oneflow.math.greater_equal(self, rhs)


def RegisterBlobOperatorTraitMethod(blob_class):
    blob_class.__add__ = __add__
    blob_class.__radd__ = __radd__
    blob_class.__sub__ = __sub__
    blob_class.__rsub__ = __rsub__
    blob_class.__mul__ = __mul__
    blob_class.__rmul__ = __rmul__
    blob_class.__truediv__ = __truediv__
    blob_class.__rtruediv__ = __rtruediv__
    blob_class.__div__ = __div__
    blob_class.__mod__ = __mod__
    blob_class.__eq__ = __eq__
    blob_class.__ne__ = __ne__
    blob_class.__lt__ = __lt__
    blob_class.__le__ = __le__
    blob_class.__gt__ = __gt__
    blob_class.__ge__ = __ge__
