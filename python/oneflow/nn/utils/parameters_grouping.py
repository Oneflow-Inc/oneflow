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


def numel_in_bucket(tensor: Tensor):
    assert flow.is_floating_point(
        tensor
    ), "params should be float tensor while grouping."

    def align(x: int, unit_size: int):
        return (x + (unit_size - 1)) // unit_size * unit_size

    # tensor memory should be align to 512 bytes for cuda operations,
    # align size depends on floating type
    return align(
        tensor.numel(),
        flow._oneflow_internal.max_alignment_size()
        // (flow.finfo(tensor.dtype).bits // 8),
    )


_tensor_or_tensors = Union[Tensor, List[Tensor], List[List[Tensor]]]

# recording all grouped params in use
_all_grouped_params = []


class ContiguousParamsGroup(object):
    def __init__(self, params_group_list: _tensor_or_tensors):
        """
        The ContiguousParamsGroup is created by 2D List of Tensors,
        which indicates the Tensors in the same 1D List should be 
        grouped into the same Tensor buffer.
        """
        assert not any(
            [any([p.is_global for p in params]) for params in params_group_list]
        ), "All parameters must be local tensor for params grouping."

        # making params_group_list 2D List of Tensors
        self.params_group_list = params_group_list.copy()
        if all(
            [
                all([isinstance(p, Tensor) for p in params])
                for params in self.params_group_list
            ]
        ):
            pass
        elif all([isinstance(p, Tensor) for p in self.params_group_list]):
            self.params_group_list = [self.params_group_list]
        elif isinstance(self.params_group_list, Tensor):
            warnings.warn("Single tensor is best not do grouping.")
            self.params_group_list = [[self.params_group_list]]
        else:
            raise ValueError("The shape of params_group_list is illegal!")

        self.grouped_tensors = []
        self.grouped_grads = []
        self.grouped_tensors_offset = {}
        self.modified_buf = set()
        last_extra_params = self._remapping_groups()
        self._parameters_grouping(last_extra_params)

    def _remapping_groups(self):
        global _all_grouped_params
        all_grouped_params_set = set(_all_grouped_params)
        last_extra_params_set = set(_all_grouped_params)

        current_buffer_tensor_mapping = collections.defaultdict(set)
        current_buffer_tensor_mapping[None] = set()
        params_requires_grad_group_list = []
        new_all_grouped_params = _all_grouped_params.copy()
        last_extra_params = []

        for params in self.params_group_list:
            params_list = []
            for p in params:
                if p.requires_grad:
                    current_buffer_tensor_mapping[p._ref_tensor].add(p)
                    self.grouped_tensors_offset[p._ref_tensor] = 0
                    params_list.append(p)

                    if p._ref_tensor is not None:
                        self.modified_buf.add(p._ref_tensor)

            params_requires_grad_group_list.append(params_list)
            params_set = set(params_list)

            # handling parameters used last time but not this time
            last_extra_params_set -= params_set

            # updating all grouped parameters set
            all_grouped_params_set |= params_set

            # keeping the parameters' order in all_grouped_params_set and last_extra_params_set
            for p in params:
                if p not in all_grouped_params_set:
                    new_all_grouped_params.append(p)
                if p in last_extra_params_set:
                    # clone last_extra_params data and detach data for memcpy
                    last_extra_params.append((p, p.detach().clone()))

        _all_grouped_params = new_all_grouped_params
        self.params_group_list = params_requires_grad_group_list

        new_params_group_list = []
        for tensors_set in current_buffer_tensor_mapping.values():
            for new_tensors in self.params_group_list:
                # handling the new groups that satisfy both conditions
                new_tensors_set = set(new_tensors)
                tensors_intersection = tensors_set & new_tensors_set

                actual_tensors_group = collections.defaultdict(list)
                # keeping the parameters' order in group list
                for tensor in new_tensors:
                    if tensor in tensors_intersection:
                        tensor_key = (tensor.dtype, tensor.device)
                        actual_tensors_group[tensor_key].append(
                            (tensor, tensor.detach().clone())
                        )

                for actual_tensors in actual_tensors_group.values():
                    if len(actual_tensors) > 0:
                        new_params_group_list.append(actual_tensors)

        self.params_group_list = new_params_group_list
        return last_extra_params

    def _parameters_grouping(self, last_extra_params):
        # handling the params used last time but not this time
        for (p, p_data) in last_extra_params:
            param_buf = p._ref_tensor

            if param_buf in self.modified_buf:
                param_grad_buf = param_buf.grad
                index = self.grouped_tensors_offset[param_buf]

                size = p.numel()
                shape = p.data.shape

                param_buf[index : index + size] = p_data.data.view(-1)
                p.data = param_buf[index : index + size].view(shape)
                p.grad = param_grad_buf[index : index + size].view(shape)

                index += numel_in_bucket(p)
                self.grouped_tensors_offset[param_buf] = index

        # handling the params used this time
        for params in self.params_group_list:
            physical_param_buf = params[0][0]._ref_tensor
            logical_bufsize = sum([numel_in_bucket(p) for (p, _) in params])

            if physical_param_buf == None:
                dtype = params[0][0].dtype
                device = params[0][0].device

                physical_param_buf = flow.zeros(
                    logical_bufsize, dtype=dtype, device=device
                )
                physical_param_grad_buf = flow.zeros(
                    logical_bufsize, dtype=dtype, device=device
                )
                physical_param_buf.grad = physical_param_grad_buf

                self.grouped_tensors_offset[physical_param_buf] = 0
            else:
                # reuse the previous allocated param_buf
                physical_param_grad_buf = physical_param_buf.grad

            index = self.grouped_tensors_offset[physical_param_buf]
            physical_index_start = index

            for (p, p_data) in params:
                size = p.numel()
                shape = p.data.shape

                physical_param_buf[index : index + size] = p_data.data.view(-1)
                p.data = physical_param_buf[index : index + size].view(shape)
                p._ref_tensor = physical_param_buf

                """
                Assigning p.grad makes p.grad_fn = <accumulate_grad>, which makes oneflow
                and pytorch behave differently as following, but won't affect this function yet.

                Oneflow:
                >>> a = oneflow.ones(1)
                >>> b = oneflow.ones(1, requires_grad=True)
                >>> b
                tensor([1.], dtype=oneflow.float32, requires_grad=True)
                >>> b.grad = a[:1]
                >>> b
                tensor([1.], dtype=oneflow.float32, grad_fn=<accumulate_grad>)

                Pytorch:
                >>> a = torch.ones(1)
                >>> b = torch.ones(1, requires_grad=True)
                >>> b
                tensor([1.], requires_grad=True)
                >>> b.grad = a[:1]
                >>> b
                tensor([1.], requires_grad=True)
                """
                p.grad = physical_param_grad_buf[index : index + size].view(shape)

                index += numel_in_bucket(p)

            self.grouped_tensors_offset[physical_param_buf] = index

            # construct the logical param_buf for new usage
            logical_param_buf = physical_param_buf[physical_index_start:index].view(
                logical_bufsize
            )
            logical_param_grad_buf = physical_param_grad_buf[
                physical_index_start:index
            ].view(logical_bufsize)
            logical_param_buf.grad = logical_param_grad_buf

            self.grouped_tensors.append(logical_param_buf)
            self.grouped_grads.append(logical_param_grad_buf)

        flow.cuda.empty_cache()

    @property
    def grouped_parameters(self):
        return self.grouped_tensors

    @property
    def grouped_parameters_grad(self):
        return self.grouped_grads
