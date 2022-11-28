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
from functools import update_wrapper
from numbers import Number
import oneflow as flow
import oneflow.nn.functional as F
from typing import Dict, Any

# NOTE(Liang Depeng): modified from
# https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py

euler_constant = 0.57721566490153286060  # Euler Mascheroni Constant


def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return flow.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs):
    eps = flow.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return flow.log(ps_clamped) - flow.log1p(-ps_clamped)
    return flow.log(ps_clamped)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
