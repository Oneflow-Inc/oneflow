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
import warnings
import oneflow as flow
import oneflow._oneflow_internal._C as _C
from oneflow.nn.module import Module


class DistributedPariticalFCSample(Module):
    """
    Sampler that can sample a 2-dim matrix along axis 0, and is always used with a fully connected layer.
    It is necessary for classification tasks with a very large number of classes such as face recognition or person re-identification.

    This module accepts two inputs, ``weight`` to be sampled and ``label`` of batch.
    And return three tensors, ``mapped_labels`` for the mapped labels after sampling, ``sampled_labels`` for the index for sampling, ``sampled_weights`` for the weight sampled.
    The sampling strategy is to sample all the positive labels and randomly sample the remaining negative labels.

    Args:
        num_sample(int): Number of classes to be sampled in the fully connected layer

    Shape:
        - weight: :math:`(C, S)`. where :math:`C` is the number of classes, :math:`S` is the feature length
        - label: :math:`(N, )`, where :math:`N` is the batch size
        - mapped_label: :math:`(N, )`, same shape as ``label``
        - sampled_label: :math:`(num\_sample, )`
        - sampled_weight: :math:`(num\_sample, S)`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> PartialFC = flow.nn.DistributedPariticalFCSample(100)
        >>> placement = flow.env.all_device_placement("cuda")
        >>> weight = flow.nn.Parameter(flow.randn(1000, 32, sbp=flow.sbp.split(0), placement=placement))
        >>> label = flow.randint(0, 1000, (64,), sbp=flow.sbp.broadcast, placement=placement)
        >>> mapped_label, sampled_label, sampled_weight = PartialFC(weight, label)
        >>> mapped_label.shape, sampled_label.shape, sampled_weight.shape
        (oneflow.Size([64]), oneflow.Size([100]), oneflow.Size([100, 32]))
        >>> flow.all(sampled_label[mapped_label] == label).item()
        True
        >>> flow.all(weight[sampled_label] == sampled_weight).item()
        True


    """

    def __init__(self, num_sample: int):
        super().__init__()
        self.num_sample = num_sample
        self._op = (
            flow.stateful_op("distributed_partial_fc_sample")
            .Input("weight")
            .Input("label")
            .Output("mapped_label")
            .Output("sampled_label")
            .Output("sampled_weight")
            .Build()
        )

    def forward(self, weight, label):
        (
            mapped_label,
            sampled_label,
            sampled_weight,
        ) = _C.dispatch_distributed_partial_fc_sample(
            self._op, weight=weight, label=label, num_sample=self.num_sample
        )
        return mapped_label, sampled_label, sampled_weight


def distributed_partial_fc_sample_op(weight, label, num_sample):
    warnings.warn(
        "oneflow.distributed_partial_fc_sample is deprecated. Please use nn.DistributedPariticalFCSample module instead.",
        DeprecationWarning,
    )
    return DistributedPariticalFCSample(num_sample)(weight, label)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
