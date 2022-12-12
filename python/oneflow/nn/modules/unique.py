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


def unique_op(input, *, return_inverse=False, return_counts=False):
    r"""
    Returns the unique elements of the input tensor.

    Args:
        input (Tensor): the input tensor
        return_inverse (bool): Whether to also return the indices for where elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique element.

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

    """
    if not return_inverse and not return_counts:
        return flow._C.unique(input)
    else:
        return flow._C.unique(input, return_inverse, return_counts)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
