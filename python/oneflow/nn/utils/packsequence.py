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
from typing import Union, Iterable

import numpy as np
import oneflow as flow

from collections import namedtuple
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module



PackedSequence_ = namedtuple('PackedSequence_',
                             ['data', 'batch_sizes', 'sorted_indices', 'unsorted_indices'])


def bind(optional, fn):
    if optional is None:
        return None
    return fn(optional)


class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``flow.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``flow.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a `:class:PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    """
    def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
        return super(PackedSequence, cls).__new__(
            cls,
            *_packed_sequence_init_args(data, batch_sizes, sorted_indices,
                                        unsorted_indices))

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").
    def pin_memory(self):
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.pin_memory(), self.batch_sizes,
                          bind(self.sorted_indices, lambda t: t.pin_memory()),
                          bind(self.unsorted_indices, lambda t: t.pin_memory()))

    def cuda(self, *args, **kwargs):
        # Tests to see if 'cuda' should be added to kwargs
        ex = flow.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(*args, device='cuda', **kwargs)

    def cpu(self, *args, **kwargs):

        ex = flow.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.device.type == 'cpu':
            return self.to(*args, **kwargs)
        return self.to(*args, device='cpu', **kwargs)

    def double(self):
        return self.to(dtype=flow.double)

    def float(self):
        return self.to(dtype=flow.float)

    def half(self):
        return self.to(dtype=flow.half)

    def long(self):
        return self.to(dtype=flow.long)

    def int(self):
        return self.to(dtype=flow.int)

    def short(self):
        return self.to(dtype=flow.short)

    def char(self):
        return self.to(dtype=flow.int8)

    def byte(self):
        return self.to(dtype=flow.uint8)

    def to(self, *args, **kwargs):
        r"""Performs dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`flow.Tensor.to`, except optional
        arguments like `non_blocking` and `copy` should be passed as kwargs,
        not args, or they will not apply to the index tensors.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`flow.dtype`
            and :class:`flow.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            # Does not forward device or dtype arg/kwargs, device is set from data.device
            kwargs = {k : v for k, v in filter(lambda t: t[0] != 'device' and t[0] != 'dtype', kwargs.items())}
            sorted_indices = bind(self.sorted_indices, lambda t: t.to(data.device, **kwargs))
            unsorted_indices = bind(self.unsorted_indices, lambda t: t.to(data.device, **kwargs))
            return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        r"""Returns true if `self.data` stored on a gpu"""
        return self.data.is_cuda

    def is_pinned(self):
        r"""Returns true if `self.data` stored on in pinned memory"""
        return self.data.is_pinned()

def _packed_sequence_init_args(data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]  # noqa: B950
    # NB: if unsorted_indices is provided, it should be the inverse permutation
    # to sorted_indices. Don't assert it here because the PackedSequence ctor
    # should only be used internally.

    if unsorted_indices is None:
        unsorted_indices = invert_permutation(sorted_indices)

    # support being called as `PackedSequence(data, batch_sizes, sorted_indices)`
    if batch_sizes is not None:
        # TODO: Re-enable this check (.type isn't supported in TorchScript)
        if batch_sizes.device.type != 'cpu':
            raise ValueError(
                "batch_sizes should always be on CPU. "
                "Instances of PackedSequence should never be created manually. "
                "They should be instantiated by functions like pack_sequence "
                "and pack_padded_sequences in nn.utils.rnn. "
                "https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence")
        return data, batch_sizes, sorted_indices, unsorted_indices

    # support being called as `PackedSequence((data, batch_sizes), *, sorted_indices)`
    else:
        assert isinstance(data, (list, tuple)) and len(data) == 2
        return data[0], data[1], sorted_indices, unsorted_indices


def _packed_sequence_init(data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> PackedSequence
    data, batch_sizes, sorted_indices, unsorted_indices = _packed_sequence_init_args(
        data, batch_sizes, sorted_indices, unsorted_indices)
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


def invert_permutation(permutation):
    # type: (Optional[Tensor]) -> Optional[Tensor]
    if permutation is None:
        return None
    output = flow.empty_like(permutation, memory_format=flow.legacy_contiguous_format)
    output.scatter_(0, permutation,
                    flow.arange(0, permutation.numel(), device=permutation.device))
    return output


def pack_padded_sequence(
    input,
    lengths,
    batch_first=False,
    enforce_sorted=False
):
    r"""Packs a Tensor containing padded sequences of variable length.

    :attr:`input` can be of size ``T x B x *`` where `T` is the length of the
    longest sequence (equal to ``lengths[0]``), ``B`` is the batch size, and
    ``*`` is any number of dimensions (including 0). If ``batch_first`` is
    ``True``, ``B x T x *`` :attr:`input` is expected.

    For unsorted sequences, use `enforce_sorted = False`. If :attr:`enforce_sorted` is
    ``True``, the sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the shortest
    one. `enforce_sorted = True` is only necessary for ONNX export.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Args:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor or list(int)): list of sequence lengths of each batch
            element (must be on the CPU if provided as a tensor).
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format.
        enforce_sorted (bool, optional): if ``True``, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            ``False``, the input will get sorted unconditionally. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    if not isinstance(lengths, flow.Tensor):
        lengths=flow.tensor(lengths, dtype=flow.int)
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = flow.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)
    data, batch_sizes = flow._C.pack_padded_sequence(input, lengths, batch_first)
    
    return _packed_sequence_init(data, batch_sizes, sorted_indices, None)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    r"""Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *``, where `T` is the length
    of the longest sequence and `B` is the batch size. If ``batch_first`` is True,
    the data will be transposed into ``B x T x *`` format.

    Example:
        >>> from flow.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> seq = flow.tensor([[1,2,0], [3,0,0], [4,5,6]])
        >>> lens = [2, 1, 3]
        >>> packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        >>> packed
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                       sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[1, 2, 0],
                [3, 0, 0],
                [4, 5, 6]])
        >>> lens_unpacked
        tensor([2, 1, 3])

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~flow.nn.Module` wrapped in :class:`~flow.nn.DataParallel`.
        See :ref:`this FAQ section <pack-rnn-unpack-with-data-parallelism>` for
        details.

    Args:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format.
        padding_value (float, optional): values for padded elements.
        total_length (int, optional): if not ``None``, the output will be padded to
            have length :attr:`total_length`. This method will throw :class:`ValueError`
            if :attr:`total_length` is less than the max sequence length in
            :attr:`sequence`.

    Returns:
        Tuple of Tensor containing the padded sequence, and a Tensor
        containing the list of lengths of each sequence in the batch.
        Batch elements will be re-ordered as they were ordered originally when
        the batch was passed to ``pack_padded_sequence`` or ``pack_sequence``.
    """

    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError("Expected total_length to be at least the length "
                             "of the longest sequence in input, but got "
                             "total_length={} and max sequence length being {}"
                             .format(total_length, max_seq_length))
        max_seq_length = total_length
    padded_output, lengths = flow._C.pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length)
    unsorted_indices = sequence.unsorted_indices
    if unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return padded_output.index_select(batch_dim, unsorted_indices), lengths[unsorted_indices]
    return padded_output, lengths


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)