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
import collections.abc
import operator
from collections import OrderedDict
from itertools import islice
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module

T = TypeVar("T")
if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
