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
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import typing
import random
import sys

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("random.gen_seed")
def api_gen_random_seed(seed: typing.Optional[int] = None):
    api = enable_if.unique([consistent_gen_random_seed, mirrored_gen_random_seed])
    return api(seed)


@enable_if.condition(hob.consistent_view_enabled)
def consistent_gen_random_seed(seed=None):
    if seed is None:
        seed = random.randint(-sys.maxsize, sys.maxsize)

    return seed, True


@enable_if.condition(hob.mirrored_view_enabled)
def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True

    return seed, has_seed
