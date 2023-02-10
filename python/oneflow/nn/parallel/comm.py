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
from collections import defaultdict
from oneflow.cuda._utils import _get_device_index, _handle_complex
from oneflow._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
    _take_tensors,
    _reorder_tensors_as,
)
from typing import List


def broadcast(tensor, devices=None, *, out=None):
    r"""Broadcasts a tensor to specified GPU devices.

    Args:
        tensor (Tensor): tensor to broadcast. Can be on CPU or GPU.
        devices (Iterable[flow.device, str or int], optional): an iterable of
          GPU devices, among which to broadcast.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing copies of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a copy of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if not ((devices is None) ^ (out is None)):
        raise RuntimeError(r"Exactly one of 'devices' and 'out'")
    if devices is not None:
        devices = [_get_device_index(d) for d in devices]
        return _comm_broadcast(tensor, devices)
    else:
        return _comm_broadcast_out(tensor, out)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcasts a sequence tensors to the specified GPUs.
    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        tensors (sequence): tensors to broadcast. Must be on the same device,
          either CPU or GPU.
        devices (Iterable[torch.device, str or int]): an iterable of GPU
          devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    devices = [_get_device_index(d) for d in devices]
    tensors = [_handle_complex(t) for t in tensors]
    return _comm_broadcast_coalesced(tensors, devices, buffer_size)


def reduce_add(inputs, destination=None):
    """Sums tensors from multiple GPUs.

    All inputs should have matching shapes, dtype, and layout. The output tensor
    will be of the same shape, dtype, and layout.

    Args:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        :attr:`destination` device.
    """
    destination = _get_device_index(destination, optional=True)
    input_size = inputs[0].size()
    root_index = None  # index of input tensor that already is on the correct device
    for i, inp in enumerate(inputs):
        assert inp.device.type != "cpu", "reduce_add expects all inputs to be on GPUs"
        if inp.get_device() == destination:
            root_index = i
        if inp.size() != input_size:
            got = "x".join(str(x) for x in inp.size())
            expected = "x".join(str(x) for x in input_size)
            raise ValueError(
                "input {} has invalid size: got {}, but expected "
                "{}".format(i, got, expected)
            )
    if root_index is None:
        raise RuntimeError(
            "reduce_add expects destination to be on the same GPU with one of the tensors"
        )

    if len(inputs) == 1:
        return inputs[0]

    # if nccl.is_available(inputs):
    #     result = torch.empty_like(inputs[root_index])
    #     nccl.reduce(inputs, output=result, root=root_index)
    # else:
    destination_device = flow.device(inputs[root_index].device.type, destination)
    nonroot = [t for i, t in enumerate(inputs) if i != root_index]
    # make a new tensor w/o clone
    result = inputs[root_index] + nonroot[0].to(device=destination_device)
    for other in nonroot[1:]:
        result.add_(other.to(device=destination_device))
    return result


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sums tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
    #       return `inputs`.
    dense_tensors: List[List] = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
    output = []
    ref_order = []

    for tensor_at_gpus in zip(*inputs):
        for coll, t in zip(dense_tensors, tensor_at_gpus):
            coll.append(t)
        ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    # now the dense ones, which have consistent sizes
    for chunks in zip(*itrs):
        flat_tensors = [
            _flatten_dense_tensors(chunk) for chunk in chunks
        ]  # (num_gpus,)
        flat_result = reduce_add(flat_tensors, destination)
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            # The unflattened tensors do not share storage, and we don't expose
            # base flat tensor anyways, so give them different version counters.
            # See NOTE [ Version Counter in comm.*_coalesced ]
            output.append(t.data)
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    """Scatters tensor across multiple GPUs.

    Args:
        tensor (Tensor): tensor to scatter. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to scatter.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
          each device. It should match :attr:`devices` in length and sums to
          ``tensor.size(dim)``. If not specified, :attr:`tensor` will be divided
          into equal chunks.
        dim (int, optional): A dimension along which to chunk :attr:`tensor`.
          Default: ``0``.
        streams (Iterable[Stream], optional): an iterable of Streams, among
          which to execute the scatter. If not specified, the default stream will
          be utilized.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results. Sizes of these tensors must match that of
          :attr:`tensor`, except for :attr:`dim`, where the total size must
          sum to ``tensor.size(dim)``.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified. When
        :attr:`out` is specified, :attr:`chunk_sizes` must not be specified and
        will be inferred from sizes of :attr:`out`.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing chunks of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a chunk of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if out is None:
        devices = [_get_device_index(d) for d in devices]
        return tuple(_comm_scatter(tensor, devices, chunk_sizes, dim, streams))
    else:
        if devices is not None:
            raise RuntimeError(
                r"'devices' must not be specified when 'out' is specified, but \
                got devices={}".format(
                    devices
                )
            )
        if chunk_sizes is not None:
            raise RuntimeError(
                r"'chunk_sizes' must not be specified when 'out' is specified, \
                but got chunk_sizes={}".format(
                    chunk_sizes
                )
            )
        return tuple(_comm_scatter_out(tensor, out, dim, streams))


def gather(tensors, dim=0, destination=None, *, out=None):
    r"""Gathers tensors from multiple GPU devices.

    Args:
        tensors (Iterable[Tensor]): an iterable of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (torch.device, str, or int, optional): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
        out (Tensor, optional, keyword-only): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.

    .. note::
        :attr:`destination` must not be specified when :attr:`out` is specified.

    Returns:
        - If :attr:`destination` is specified,
            a tensor located on :attr:`destination` device, that is a result of
            concatenating :attr:`tensors` along :attr:`dim`.
        - If :attr:`out` is specified,
            the :attr:`out` tensor, now containing results of concatenating
            :attr:`tensors` along :attr:`dim`.
    """
    tensors = [_handle_complex(t) for t in tensors]
    if out is None:
        if destination == -1:
            warnings.warn(
                "Using -1 to represent CPU tensor is deprecated. Please use a "
                'device object or string instead, e.g., "cpu".'
            )
        destination = _get_device_index(destination, allow_cpu=True, optional=True)
        return _comm_gather(tensors, dim, destination)
    else:
        if destination is not None:
            raise RuntimeError(
                "'destination' must not be specified when 'out' is specified, but "
                "got destination={}".format(destination)
            )
        return _comm_gather_out(tensors, out, dim)


def _comm_scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None):
    """
    Scatters tensor across multiple GPUs.

    Args:
        tensor (Tensor): tensor to scatter. Can be on CPU or GPU.
        devices (Iterable[flow.device, str or int], optional): an iterable of
          GPU devices, among which to scatter.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
          each device. It should match :attr:`devices` in length and sums to
          ``flow.size(dim)``. If not specified, :attr:`tensor` will be divided
          into equal chunks.
        dim (int, optional): A dimension along which to chunk :attr:`tensor`.
          Default: ``0``.
        streams (Iterable[Stream], optional): an iterable of Streams, among
          which to execute the scatter. If not specified, the default stream will
          be utilized.
    """
    if len(devices) < 1:
        raise RuntimeError(r"Expected at least one device to scatter to")
    if chunk_sizes is not None:
        if len(chunk_sizes) != len(devices):
            raise RuntimeError(r"Expected devices and chunk_sizes to be of same length")
    out_tensors = (
        flow.split(tensor, chunk_sizes, dim=dim)
        if chunk_sizes
        else flow.chunk(tensor, len(devices), dim=dim)
    )
    tensor_index = -1 if tensor.device.type == "cpu" else tensor.get_device()
    out_tensors = list(out_tensors)
    for index in range(len(out_tensors)):
        device_index = devices[index]
        if device_index != tensor_index:
            out_tensors[index] = out_tensors[index].cuda(device_index)
    return out_tensors


def _comm_scatter_out(tensor, out_tensors, dim, streams):
    """
    Scatters tensor across multiple GPUs.

    Args:
        tensor (Tensor): tensor to scatter. Can be on CPU or GPU.
        out_tensor (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results. Sizes of these tensors must match that of
          :attr:`flow`, except for :attr:`dim`, where the total size must
          sum to ``flow.size(dim)``.
        dim (int, optional): A dimension along which to chunk :attr:`tensor`.
          Default: ``0``.
        streams (Iterable[Stream], optional): an iterable of Streams, among
          which to execute the scatter. If not specified, the default stream will
          be utilized.
    """
    if len(out_tensors) < 1:
        raise RuntimeError(r"Expected at least one output tensor to scatter to")

    chunk_sizes = []
    total_size = 0
    for index in range(len(out_tensors)):
        total_size += out_tensors[index].size(dim)
        chunk_sizes.append(out_tensors[index].size(dim))
        out_tensors_size = list(out_tensors[index].size())
        out_tensors_size[dim] = tensor.size(dim)
        if not out_tensors[index].is_cuda:
            raise RuntimeError(
                r"Expected all output tensors to be CUDA tensors, but output tensor at index {}".format(
                    index
                )
            )
        if out_tensors[index].dim() != tensor.dim():
            raise RuntimeError("Output tensor at index 0 has incorrect shape")
        if list(tensor.size()) != out_tensors_size:
            raise RuntimeError(
                r"Output tensor at index 0 has incorrect shape".format(index)
            )
    if total_size != tensor.size(dim):
        raise RuntimeError(
            r"Total size for output tensors along scatter dim does not match"
        )

    chunks = flow.split(tensor, chunk_sizes, dim=dim)
    for index in range(len(out_tensors)):
        out_tensors[index].copy_(chunks[index])
    return out_tensors


def _comm_broadcast(tensor, devices):
    """
    Broadcast tensor to specified GPU devices.

    Args:
        tensor(Tensor):tensor to broadcast.
        devices(List):the index of GPU devices.

    Return:
        A list containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    if tensor.device.type == "cpu":
        tensor = tensor.cuda()
    diff_device_dst_tensors = list()
    for index in devices:
        assert index >= 0, "Expected non-negative device index, but got ".format(index)
        if index != tensor.get_device():
            diff_device_dst_tensors.append(
                flow.empty(
                    tensor.size(), dtype=tensor.dtype, device=flow.device("cuda", index)
                )
            )
    _boradcast_out_impl(tensor, diff_device_dst_tensors)
    dst_tensors = list()
    i = 0
    for index in devices:
        if index != tensor.get_device():
            dst_tensors.append(diff_device_dst_tensors[i])
            i += 1
        else:
            dst_tensors.append(tensor)
    return dst_tensors


def _comm_broadcast_out(tensor, out_tensor):
    """
    Broadcast tensor to specified out_tensor.

    Args:
        tensor(Tensor):tensor to broadcast.
        out_tensor(List):a list contains output tensor.
    """
    for index in range(len(out_tensor)):
        if not out_tensor[index].is_cuda:
            raise RuntimeError(
                r"Expected all output tensors to be CUDA tensors, but output tensor at index {}".format(
                    index
                )
            )
        if out_tensor[index].size() != tensor.size():
            raise RuntimeError(
                r"Expected all output tensors to have same shape as the source at index {}".format(
                    index
                )
            )
    return _boradcast_out_impl(tensor, out_tensor)


def _boradcast_out_impl(tensor, out_tensor):
    """
    Copy source tensor to output tensor.
    Args:
        tensor(Tensor):the source tensor to broadcast.
        out_tensor(List):the list of containing copies of source tensor.
    Returns:
        A list containing tensor, every element is the same as source tensor.
    """
    for index in range(len(out_tensor)):
        out_tensor[index].copy_(tensor)
    return out_tensor


def _comm_broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcasts a sequence tensors to the specified GPUs.
    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        tensors (List): tensors to broadcast. Must be on the same device GPU.
        devices (List[int]): an list of GPU devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    # check elements of tensors should be same device
    if not all(param.get_device() == devices[0] for param in tensors):
        raise RuntimeError(r"Must be on the same device GPU")
    tensorlist2d = [[] for _ in range(len(devices))]
    tensorlist2d[0] = tensors
    for data in tensors:
        results = _comm_broadcast(tensor=data, devices=devices)
        for i, device_tensor in enumerate(results):
            if i == 0:
                continue
            tensorlist2d[i].append(device_tensor)
    return tuple(tensorlist2d)


def _comm_gather(tensors, dim=0, destination=None):
    r"""
    Args:
        tensors (List): an list of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (torch.device, str, or int, optional): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
    """
    pass


def _comm_gather_out(tensors, out, dim):
    r"""
    Args:
        tensors (List): an list of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        out (Tensor): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
    """
    if len(tensors) < 1:
        raise RuntimeError(r"Expected at least one tensor to gather from")
    expected_size = list(tensors[0].size())
    total_size = 0
    for index in range(len(tensors)):
        if not tensors[index].is_cuda:
            raise RuntimeError(r"Expected all input tensors to be CUDA tensors, ")
        if tensors[index].ndim != len(expected_size):
            raise RuntimeError(
                r"Expected all input tensors to have the same number of dimensions"
            )
        expected_size[dim] = tensors[index].size(dim)
        for dimension in range(len(expected_size)):
            if expected_size[dimension] != tensors[index].size(dimension):
                raise RuntimeError(
                    r"Input tensor at index {} has invalid shape".format(dimension)
                )

        total_size += tensors[index].size(dim)
    expected_size[dim] = total_size
    if expected_size != list(out.size()):
        raise RuntimeError(r"Out tensor shape don't match")

    return _comm_gather_out_impl(tensors, out, dim)


def _comm_gather(tensors, dim, destination):
    r"""
    Args:
        tensors (List): an list of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (int): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
    """
    if len(tensors) < 1:
        raise RuntimeError(r"Expected at least one tensor to gather from")
    expected_size = list(tensors[0].size())
    total_size = 0
    for index in range(len(tensors)):
        if not tensors[index].is_cuda:
            raise RuntimeError(r"Expected all input tensors to be CUDA tensors, ")
        if tensors[index].ndim != len(expected_size):
            raise RuntimeError(
                r"Expected all input tensors to have the same number of dimensions"
            )
        expected_size[dim] = tensors[index].size(dim)
        for dimension in range(len(expected_size)):
            if expected_size[dimension] != tensors[index].size(dimension):
                raise RuntimeError(
                    r"Input tensor at index {} has invalid shape".format(index)
                )

        total_size += tensors[index].size(dim)
    expected_size[dim] = total_size
    device = (
        flow.device("cuda", destination) if destination != -1 else flow.device("cpu")
    )
    result = flow.empty(expected_size, device=device)
    return _comm_gather_out_impl(tensors, result, dim)


def _comm_gather_out_impl(tensors, out_tensor, dim):
    r"""
    Args:
        tensors (List): an list of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        out (Tensor): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
    """
    # chunk_sizes = list()
    # for t in tensors:
    #     chunk_sizes.append(t.size(dim))
    # chunks = flow.split(out_tensor, chunk_sizes, dim = dim)
    # for index in range(len(chunks)):
    #     chunks[index].copy_(tensors[index])
    tensors = [t.cpu() for t in tensors]
    tmp = flow.cat(tensors, dim=dim)
    out_tensor.copy_(tmp)
    return out_tensor
