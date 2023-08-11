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
import torch
import oneflow
import numpy as np

input_arr = np.array(
    [
        [-0.94630778, -0.83378579, -0.87060891],
        [2.0289922, -0.28708987, -2.18369248],
        [0.35217619, -0.67095644, -1.58943879],
        [0.08086036, -1.81075924, 1.20752494],
        [0.8901075, -0.49976737, -1.07153746],
        [-0.44872912, -1.07275683, 0.06256855],
        [-0.22556897, 0.74798368, 0.90416439],
        [0.48339456, -2.32742195, -0.59321527],
    ],
    dtype=np.float32,
)
x = torch.tensor(input_arr, device="cuda")

def fn(x):
    y = torch.relu(x)
    return y

jit_mod = torch.jit.trace(fn, x)
print(jit_mod)
