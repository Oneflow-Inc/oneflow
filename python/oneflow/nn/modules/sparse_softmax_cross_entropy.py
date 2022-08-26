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
import oneflow as flow


def sparse_softmax_cross_entropy(labels, logits):
    """The interface is consistent with TensorFlow.    
    The documentation is referenced from: 
    https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    
    Computes sparse softmax cross entropy between `logits` and `labels`.

    Measures the probability error in discrete classification tasks in which the
    classes are mutually exclusive (each entry is in exactly one class).  For
    example, each CIFAR-10 image is labeled with one and only one label: an image
    can be a dog or a truck, but not both.

    A common use case is to have logits of shape
    `[batch_size, num_classes]` and have labels of shape
    `[batch_size]`, but higher dimensions are supported, in which
    case the `dim`-th dimension is assumed to be of size `num_classes`.
    `logits` must have the dtype of `float16`, `float32`, or `float64`, and
    `labels` must have the dtype of `int32` or `int64`.

    Args:
        labels (Tensor): shape with [d_0, d_1, ..., d_{r-1}] (where `r` is rank of
            `labels` and output) and dtype `int32` or `int64`. Each entry in `labels`
            must be an index in [0, num_classes).
        logits (Tensor): Per-label activations (typically a linear output) of shape
            [d_0, d_1, ..., d_{r-1}, num_classes] and dtype `float16`, `float32`, or
            `float64`. These activation energies are interpreted as unnormalized log
            probabilities.

    Returns:
        output (Tensor): A `Tensor` of the same shape as `labels` and of the same type as `logits`
        with the softmax cross entropy loss.

    Examples::
        >>> import numpy as np
        >>> import oneflow as flow
        >>> np_logits = np.array(
        ...      [
        ...          [2.0, -5.0, 0.5, -0.1],
        ...          [0.0, 0.0, 1.9, 1.4],
        ...          [-100.0, 100.0, -100.0, -100.0],
        ...      ]
        ...  )
        >>> np_labels = np.array([0, 3, 1])
        >>> logits = flow.tensor(np_logits, dtype=flow.float32)
        >>> labels = flow.tensor(np_labels, dtype=flow.int32)
        >>> output = flow.nn.functional.sparse_softmax_cross_entropy(
        ...     labels=labels, logits=logits
        ... )
        >>> output
        tensor([ 2.9751e-01,  1.1448e+00, -1.4305e-06], dtype=oneflow.float32)
    """
    return flow._C.sparse_softmax_cross_entropy(logits, labels)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
