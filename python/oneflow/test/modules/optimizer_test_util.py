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
    np_grad_is_list = True
    if isinstance(np_grad, np.ndarray):
        np_grad_is_list = False
        np_grad = [np_grad]

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = np.max(np.abs(np_grad))
    elif norm_type == float("-inf"):
        total_norm = np.min(np.abs(np_grad))
    else:
        norms = np_grad
        total_norm = []
        for i, norm in enumerate(norms):
            for j in range(np_grad[i].ndim, 0, -1):
                norm = np.linalg.norm(norm, norm_type, axis=j - 1)
            total_norm.append(norm)
        total_norm = np.linalg.norm(np.array(total_norm, dtype=np.float32), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in np_grad:
            grad *= clip_coef

    if not np_grad_is_list:
        np_grad = np_grad[0]
    return total_norm, np_grad
