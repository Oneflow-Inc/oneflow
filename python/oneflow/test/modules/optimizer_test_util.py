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


def clip_grad_norm_np(np_grad, max_norm, norm_type):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = np.max(np.abs(np_grad))
    if norm_type == float("-inf"):
        total_norm = np.min(np.abs(np_grad))
    elif norm_type == 0:
        total_norm = np.sum(np.stack([np.sum(np_grad != 0)]) != 0)
    else:
        total_norm = np_grad
        for i in range(np_grad.ndim, 0, -1):
            total_norm = np.linalg.norm(total_norm, norm_type, axis=i - 1)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        np_grad = np_grad * clip_coef
    return total_norm, np_grad
