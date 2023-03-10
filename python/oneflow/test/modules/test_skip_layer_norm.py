import os
import numpy as np
import unittest

import oneflow as flow
import oneflow.unittest
import torch

def _skip_layer_norm(x, normalized_shape, gamma=None, beta=None, bias=None, skip=None, alpha=1e-5, eps=1e-6):
    return flow._C.skip_layer_norm(
        x=x,
        gamma=gamma,
        beta=beta,
        bias=bias,
        skip=skip,
        alpha=alpha,
        epsilon=eps,
    )

# TODO: 