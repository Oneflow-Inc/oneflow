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
import collections
import warnings
from typing import Union, List

import oneflow as flow
from oneflow.framework.tensor import Tensor


_tensor_or_tensors = Union[Tensor, List[Tensor], List[List[Tensor]]]


def numel_in_bucket(tensor: Tensor):
    assert flow.is_floating_point(tensor), "params grouping only support float tensor."

    def align(x: int, unit_size: int):
        return (x + (unit_size - 1)) // unit_size * unit_size

    # tensor memory should be align to 512 bytes for cuda operations,
    # align size depends on floating type
    return align(
        tensor.numel(),
        flow._oneflow_internal.max_alignment_size()
        // (flow.finfo(tensor.dtype).bits // 8),
    )


class ContiguousParamsGroup(object):
    """Arange tensors into contiguous buffer according to their group.

    Args:
        params_group_list (Iterable[Tensor] or Tensor): an iterable of Tensors or
            a single Tensor that will be made into buffers.
        group_on_current_buffer (bool, optional): whether to group tensors on allocated
            buffers. (default: True)

    Note:
        The ContiguousParamsGroup is created by 2D List of Tensors, which indicates the
        Tensors in the same 1D List should be grouped into the same Tensor buffer, otherwise
        try to make them into a 2D List.

        If group_on_current_buffer is set True but there is not any buffer created before,
        ContiguousParamsGroup will allocate default buffers for all parameters.
    """

    def __init__(
        self,
        params_group_list: _tensor_or_tensors,
        group_on_current_buffer: bool = True,
    ):
        self.params_group_list = params_group_list.copy()

        self._make_valid_params_group_list()
        self._remove_no_grad_tensors()
        self._check_tensor_position_consistency()

        self.group_on_current_buffer = group_on_current_buffer
        self.grouped_tensors = []
        self.grouped_grads = []

        if not self.group_on_current_buffer:
            self._parameters_grouping_on_new_buffer()
        else:
            self._check_current_buffer()
            self._parameters_grouping_on_current_buffer()

    def _make_valid_params_group_list(self):
        """making params_group_list 2D List of Tensors
        """
        if isinstance(self.params_group_list, Tensor):
            warnings.warn("Single tensor is best not do grouping.")
            self.params_group_list = [[self.params_group_list]]
        elif all([isinstance(p, Tensor) for p in self.params_group_list]):
            self.params_group_list = [self.params_group_list]
        elif all(
            [
                all([isinstance(p, Tensor) for p in params])
                for params in self.params_group_list
            ]
        ):
            pass
        else:
            raise ValueError("The shape of params_group_list is illegal!")

    def _remove_no_grad_tensors(self):
        self.params_group_list = [
            [p for p in params if p.requires_grad] for params in self.params_group_list
        ]

    def _check_tensor_position_consistency(self):
        if all(
            [all([p.is_global for p in params]) for params in self.params_group_list]
        ):
            self.is_global = True
        elif all(
            [all([p.is_local for p in params]) for params in self.params_group_list]
        ):
            self.is_global = False
        else:
            raise ValueError(
                "Parameters must be all local tensors or all global tensors for params grouping."
            )

    def _check_current_buffer(self):
        """If all tensors are not held by any buffer, try to create buffer.
        """
        for params in self.params_group_list:
            for p in params:
                if p._ref_tensor is not None:
                    return

        warnings.warn("create defualt buffer for all parameters as one group.")

        self._physical_preparation = ContiguousParamsGroup(
            self.params_group_list, group_on_current_buffer=False,
        )

    def _make_buffer_params_mapping(self):
        buffer_params_mapping = collections.defaultdict(list)

        for params in self.params_group_list:
            for p in params:
                if p._ref_tensor is not None:
                    assert (
                        p._ref_index < p._ref_tensor.numel()
                    ), "invalid ref tensor index."
                    buffer_params_mapping[p._ref_tensor].append((p._ref_index, p))

        for buffer, params_list in buffer_params_mapping.items():
            buffer_params_mapping[buffer] = sorted(params_list, key=lambda x: x[0])

        return buffer_params_mapping

    def _parameters_grouping_on_new_buffer(self):
        # Use the group in params_group_list to create default buffer.
        # A buffer that is too large will affect the parallelism of different parameters.

        params_buffer_size = {}
        physical_params_buffer = {}
        params_buffer_index = {}

        for idx, params in enumerate(self.params_group_list):
            for p in params:
                if self.is_global:
                    tensor_key = (p.dtype, p.placement, p.sbp, idx)
                else:
                    tensor_key = (p.dtype, p.device, idx)

                params_buffer_size[tensor_key] = params_buffer_size.get(
                    tensor_key, 0
                ) + numel_in_bucket(p)

        for tensor_key, buffer_size in params_buffer_size.items():
            dtype = tensor_key[0]

            if self.is_global:
                placement = tensor_key[1]
                sbp = tensor_key[2]
                physical_param_buf = flow.zeros(
                    buffer_size, dtype=dtype, placement=placement, sbp=sbp
                )
                physical_param_buf.grad = flow.zeros(
                    buffer_size, dtype=dtype, placement=placement, sbp=sbp
                )
            else:
                device = tensor_key[1]
                physical_param_buf = flow.zeros(buffer_size, dtype=dtype, device=device)
                physical_param_buf.grad = flow.zeros(
                    buffer_size, dtype=dtype, device=device
                )

            self.grouped_tensors.append(physical_param_buf)
            self.grouped_grads.append(physical_param_buf.grad)
            physical_params_buffer[tensor_key] = physical_param_buf
            params_buffer_index[tensor_key] = 0

        for idx, params in enumerate(self.params_group_list):
            for p in params:
                if self.is_global:
                    tensor_key = (p.dtype, p.placement, p.sbp, idx)
                else:
                    tensor_key = (p.dtype, p.device, idx)

                param_buf = physical_params_buffer[tensor_key]
                index = params_buffer_index[tensor_key]
                size = p.numel()
                shape = p.data.shape

                assert index + numel_in_bucket(p) <= param_buf.numel()

                param_buf[index : index + size] = p.data.detach().clone().view(-1)
                p.data = param_buf[index : index + size].view(shape)
                p.grad = param_buf.grad[index : index + size].view(shape)

                p._ref_tensor = param_buf
                p._ref_index = index

                index += numel_in_bucket(p)
                params_buffer_index[tensor_key] = index

    def _parameters_grouping_on_current_buffer(self):
        buffer_params_mapping = self._make_buffer_params_mapping()

        if buffer_params_mapping is None or len(buffer_params_mapping) == 0:
            warnings.warn(
                "Since nn.Module didn't use make_contiguous_params_group() to create "
                "a contiguous module, the remapping won't make any difference for parameters. "
            )

        params_group = []
        for params in self.params_group_list:
            group = set()
            for p in params:
                group.add(p)
            params_group.append(group)

        # handling the parameters already on allocated buffers
        # try best to make the adjacent tensors on device into same logical buffer
        for param_buf, params in buffer_params_mapping.items():
            logical_buffer_start, logical_buffer_size = 0, 0
            pre_group_index = -1
            params_cnt = len(params)

            for p_index, (_, p) in enumerate(params):
                current_group_index = -1

                for group_index, group in enumerate(params_group):
                    if p in group:
                        current_group_index = group_index
                        break

                if current_group_index == -1:
                    continue

                params_group[current_group_index].remove(p)

                def _make_logical_buf():
                    nonlocal logical_buffer_start, logical_buffer_size
                    nonlocal pre_group_index, current_group_index

                    pre_group_index = current_group_index

                    if logical_buffer_size == 0:
                        return

                    logical_param_buf = param_buf[
                        logical_buffer_start : logical_buffer_start
                        + logical_buffer_size
                    ].view(logical_buffer_size)
                    logical_param_grad_buf = param_buf.grad[
                        logical_buffer_start : logical_buffer_start
                        + logical_buffer_size
                    ].view(logical_buffer_size)
                    logical_param_buf.grad = logical_param_grad_buf

                    self.grouped_tensors.append(logical_param_buf)
                    self.grouped_grads.append(logical_param_grad_buf)

                    logical_buffer_start += logical_buffer_size
                    logical_buffer_size = 0

                if current_group_index != pre_group_index:
                    _make_logical_buf()

                logical_buffer_size += numel_in_bucket(p)

                if p_index == params_cnt - 1:
                    _make_logical_buf()

        # handling params not on any buffer
        # however, we don't make new tensors into contiguous buffer this time
        for group in params_group:
            for p in group:
                self.grouped_tensors.append(p)
                self.grouped_grads.append(p.grad)

    @property
    def grouped_parameters(self):
        """the grouped contiguous parameters
        """
        return self.grouped_tensors

    @property
    def grouped_parameters_grad(self):
        """the grouped contiguous parameters' gradient
        """
        return self.grouped_grads
