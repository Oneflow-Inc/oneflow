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
from . import comm
from oneflow.autograd import Function
from oneflow.cuda._utils import _get_device_index
from typing import List, Optional


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(
            i.device.type != "cpu" for i in inputs
        ), "Broadcast function not implemented for CPU tensors"
        # convert tensor to list
        target_gpus = [t.item() for t in list(target_gpus)]
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return tuple()
        ctx.needs_input_grad = (False, True)
        ctx.num_inputs = len(inputs)
        ctx.input_device = inputs[0].get_device()
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, ) + ReduceAddCoalesced.apply(
            flow.tensor(ctx.input_device), flow.tensor(ctx.num_inputs), *grad_outputs
        )


class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        destination = destination.item()
        num_inputs = num_inputs.item()
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]
        grads_ = [grads[i : i + num_inputs] for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads_, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, ) + Broadcast.apply(flow.tensor(ctx.target_gpus), *grad_outputs)


class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(
            i.device.type != "cpu" for i in inputs
        ), "Gather function not implemented for CPU tensors"
        # using -1 represents 'cpu' device
        target_device = target_device.item()
        target_device = _get_device_index(target_device, True)
        ctx.target_device = target_device
        ctx.dim = dim.item()
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim.item() == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector."
            )
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(dim.item()) for i in inputs)
        return comm.gather(inputs, dim.item(), target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(
            flow.tensor(ctx.input_gpus), flow.tensor(ctx.input_sizes), flow.tensor(ctx.dim), grad_output
        )
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [t.item() for t in list(target_gpus)]
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus_len = len(target_gpus)
        # use flow.tensor(-1) represent None
        if chunk_sizes.nelement() == 1 and chunk_sizes.item() == -1:
            chunk_sizes = None
        else:
            chunk_sizes = [t.item() for t in list(chunk_sizes)]
        ctx.dim = dim.item()
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(flow.tensor(ctx.input_device), flow.tensor(ctx.dim), *grad_output)


# background streams used for copying
# _streams: Optional[List[Optional[torch.cuda.Stream]]] = None


# def _get_stream(device: int):
#     """Gets a background stream for copying between CPU and GPU"""
#     global _streams
#     if device == -1:
#         return None
#     if _streams is None:
#         _streams = [None] * torch.cuda.device_count()
#     if _streams[device] is None:
#         _streams[device] = torch.cuda.Stream(device)
#     return _streams[device]
