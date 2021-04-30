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
import sys
import random
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.id_util as id_util


@oneflow_export("nn.Dropout")
class Dropout(Module):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow

        m = flow.nn.Dropout(p=0.5)
        arr = np.array(
            [
                [-0.7797, 0.2264, 0.2458, 0.4163],
                [0.4299, 0.3626, -0.4892, 0.4141],
                [-1.4115, 1.2183, -0.5503, 0.6520],
            ]
        )
        x = flow.Tensor(arr)
        y = m(x)
        # likely output:
        # [[-0.      0.      0.4916  0.8326]
        # [ 0.8598  0.     -0.      0.8282]
        # [-2.823   2.4366 -0.      1.304 ]]

    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        assert inplace is False, "Not support inplace=True yet!"
        assert 0 <= p < 1.0, "Dropout rate should be in range [0,1)"
        self.rate = p
        seed = random.randint(-sys.maxsize, sys.maxsize)
        scale = float(1.0 / (1.0 - self.rate))

        self._op = (
            flow.builtin_op("dropout")
            .Input("in")
            .Input("mask")
            .Output("out")
            .Attr("scale", scale)
            .Build()
        )
        self._mask_op = (
            flow.builtin_op("random_mask_like")
            .Input("like")
            .Output("out")
            .Attr("rate", self.rate)
            .Attr("seed", seed)
            .Build()
        )

    def forward(self, x):
        if self.rate == 0.0:
            return x
        mask = self._mask_op(x)[0]
        return self._op(x, mask)[0]
