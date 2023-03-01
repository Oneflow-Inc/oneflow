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


def unique_op(
    input, sorted=True, return_inverse=False, return_counts=False, dtype=flow.int
):
    r"""
    Returns the unique elements of the input tensor.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.unique.html.

    Args:
        input (Tensor): The input tensor.
        sorted (bool): Whether to sort the unique elements in ascending order before returning as output.
        return_inverse (bool): Whether to also return the indices for where elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique element.
        dtype (flow.dtype): Dtype of the returned indices and counts.

    Returns:
        oneflow.Tensor or List of oneflow.Tensor:

        - **output** (Tensor): the output list of unique scalar elements.

        - **inverse_indices** (Tensor): (optional) if return_inverse is True, 
          there will be an additional returned tensor (same shape as input) representing
          the indices for where elements in the original input map to in the output;
          otherwise, this function will only return a single tensor.

        - **counts** (Tensor): (optional) if return_counts is True, there will be an additional
          returned tensor (same shape as output or output.size(dim), if dim was specified)
          representing the number of occurrences for each unique value or tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([3, 1, 2, 0 ,2])
        >>> flow.unique(x)
        tensor([0, 1, 2, 3], dtype=oneflow.int64)
        >>> flow.unique(x, sorted=False)
        tensor([3, 1, 2, 0], dtype=oneflow.int64)
        >>> results, indices = flow.unique(x, return_inverse=True)
        >>> indices
        tensor([3, 1, 2, 0, 2], dtype=oneflow.int32)
        >>> results, counts = flow.unique(x, return_counts=True)
        >>> counts
        tensor([1, 1, 2, 1], dtype=oneflow.int32)
        >>> results, indices = flow.unique(x, return_inverse=True, dtype=flow.long)
        >>> indices
        tensor([3, 1, 2, 0, 2], dtype=oneflow.int64)

    """
    if not return_inverse and not return_counts:
        return flow._C.unique(input, sorted, dtype=dtype)
    else:
        return flow._C.unique(
            input,
            sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dtype=dtype,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
