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
import math

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.init import _calculate_fan_in_and_fan_out
from oneflow.nn.module import Module
from typing import Tuple 


class FusedMLP(Module):
    """Applies a linear transformation with relu activation to the incoming data: :math:`y = ReLU(xA^T + b)`

    Currently only support in GPU. 

    Args:
        in_features: size of each input sample

        hidden_features_lists: A tuple of each Linear layer output size

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = {in\\_features}`

        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = {out\\_features}`.

    Attr:
        - :attr:`weight`: the learnable weights of the module of shape :math:`({out\\_features}, {in\\_features})`. The values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where :math:`(k = 1 / {in\\_features})`

        - :attr:`bias`: the learnable bias of the module of shape :math:`({out\\_features})`. If :attr:`bias` is ``True``, the values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where :math:`(k = 1 / {in\\_features})`


    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        

        >>> m = flow.nn.FusedMLP(20, 30).to("cuda")
        >>> input = flow.Tensor(np.random.randn(128, 20)).to("cuda")
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([128, 30])

    """

    def __init__(self, in_features: int, hidden_features_lists: Tuple[int], out_features: int, skip_last_activation=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features_lists = hidden_features_lists
        self.out_features = out_features
        # TODO(zzk): Add more activation support.
        self.skip_last_activation = skip_last_activation 
        self.weights = []
        self.biases = []
        self.hidden_layer_num = len(hidden_features_lists)
        assert (self.hidden_layer_num + 1) % 2 == 0, "Currently only support even times Dense Layers. "
        self.fuse_layer_num = (self.hidden_layer_num + 1) // 2
        """
        in = 128
        hidden = [64, 32, 16, 32, 16]
        out = 4
        (128, 64), (64, 32) (32, 16) (64, 32) (32, 16) (16, 4)
        self.fuse_layer_num = (len(hidden_features_lists) + 1) // 2 = 6
        """ 
        # First layer. 
        self.add_parameters(in_features, hidden_features_lists[0], 0)
        # Middle Layer. 
        for idx in range(self.hidden_layer_num-1): 
            print(idx)
            self.add_parameters(hidden_features_lists[idx], hidden_features_lists[idx+1], idx+1)
        # Last layer. 
        self.add_parameters(hidden_features_lists[-1], out_features, self.hidden_layer_num)
        
        for idx, weight in enumerate(self.weights):
            self.register_parameter(f'weight_{idx}', weight)
        for idx, bias in enumerate(self.biases):
            self.register_parameter(f'bias_{idx}', bias)
        
        self.reset_parameters()

    def add_parameters(self, in_features, out_features, layer_idx) -> None: 
        self.weights.append(flow.nn.Parameter(flow.Tensor(out_features, in_features)))
        self.biases.append(flow.nn.Parameter(flow.Tensor(out_features)))


    def reset_parameters(self) -> None:
        for layer_idx in range(self.hidden_layer_num + 1): 
            flow.nn.init.kaiming_uniform_(self.weights[layer_idx], a=math.sqrt(5))
            (fan_in, _) = _calculate_fan_in_and_fan_out(self.weights[layer_idx])
            bound = 1 / math.sqrt(fan_in)
            flow.nn.init.uniform_(self.biases[layer_idx], -bound, bound)


    def forward(self, x):
        # First Layer. 
        res = flow._C.relu(flow._C.fused_matmul_bias_add_relu(
            x, 
            self.weights[0], self.biases[0], 
            self.weights[1], self.biases[1], 
            alpha1=1.0, alpha2=1.0
        ))
        for idx in range(1, (self.fuse_layer_num)//2 - 1, 1): 
        # for idx in range(1, (self.hidden_layer_num), 1): 
            print(idx)
            res = flow._C.relu(flow._C.fused_matmul_bias_add_relu(
                res, 
                self.weights[idx*2], self.biases[idx*2], 
                self.weights[idx*2+1], self.biases[idx*2+1], 
                alpha1=1.0, alpha2=1.0
            ))
        # Last Layer. 
        res = flow._C.fused_matmul_bias_add_relu(
                res, 
                self.weights[self.hidden_layer_num-1], self.biases[self.hidden_layer_num-1], 
                self.weights[self.hidden_layer_num], self.biases[self.hidden_layer_num], 
                alpha1=1.0, alpha2=1.0
            )
        if not self.skip_last_activation: 
            flow._C.relu(res)
        return res

    def extra_repr(self) -> str:
        return "in_features={}, hidden_features_lists={}, bias={}".format(
            self.in_features, self.hidden_features_lists, self.bias is not None
        )
    

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
