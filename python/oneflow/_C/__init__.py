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
from oneflow._oneflow_internal._C import *
import oneflow._C._nn as _nn
import warnings


def allclose(input, other, atol=1e-08, rtol=1e-05, equal_nan=False):
    return isclose(input, other, atol, rtol, equal_nan).all().item()


def _log_api_usage_once(event):
    warnings.warn("_log_api_usage_once is not implemented in oneflow")
