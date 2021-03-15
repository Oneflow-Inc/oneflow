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
import numpy as np


def forward(args):
    print("user sigmoid forward args", args)
    (x,) = args
    y = 1 / (1 + np.exp(-x))
    return y


def backward(args):
    print("user sigmoid backward args", args)
    y, dy = args
    return y * (1 - y) * dy
