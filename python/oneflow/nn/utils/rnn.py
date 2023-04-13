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
from audioop import reverse
from collections import namedtuple
from typing import List, Tuple, Union, Iterable, Optional
import warnings

import oneflow as flow
from oneflow.framework.tensor import Tensor

# The implementation of rnn util is modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/rnn.py


def bind(optional, fn):
    if optional is None:
        return None
    return fn(optional)


def invert_permutation(permutation: Optional[Tensor]) -> Optional[Tensor]:
    if permutation is None:
        return None
    return flow.scatter(
        flow.zeros_like(permutation),
        0,
        permutation,
        flow.arange(
            0, permutation.numel(), device=permutation.device, dtype=flow.int32
        ),
    )


class PackedSequence(object):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.utils.rnn.PackedSequence.html.
    
    Holds the data and list of :attr:`batch_sizes` of a packed sequence.

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
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``oneflow.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``oneflow.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a `:class:PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    """

    def __init__(
        self,
        data: Tensor,
        batch_sizes: Optional[Tensor] = None,
        sorted_indices: Optional[Tensor] = None,
        unsorted_indices: Optional[Tensor] = None,
    ):
        self.sorted_indices = sorted_indices
        if unsorted_indices is None:
            self.unsorted_indices = invert_permutation(sorted_indices)
        self.sorted_indices = sorted_indices

        if batch_sizes is not None:
            if batch_sizes.device.type != "cpu":
                raise ValueError(
                    "batch_sizes should always be on CPU. "
                    "Instances of PackedSequence should never be created manually. "
                    "They should be instantiated by functions like pack_sequence "
                    "and pack_padded_sequences in nn.rnn_utils "
                )
            self.data = data
            self.batch_sizes = batch_sizes
        else:
            assert isinstance(data, (list, tuple)) and len(data) == 2
            self.data = data[0]
            self.batch_sizes = data[1]

    def pin_memory(self):
        return PackedSequence(
            self.data.pin_memory(),
            self.batch_sizes,
            bind(self.sorted_indices, lambda t: t.pin_memory()),
            bind(self.unsorted_indices, lambda t: t.pin_memory()),
        )

    def cuda(self, *args, **kwargs):
        ex = flow.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(*args, device="cuda", **kwargs)

    def cpu(self, *args, **kwargs):

        ex = flow.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.device.type == "cpu":
            return self.to(*args, **kwargs)
        return self.to(*args, device="cpu", **kwargs)

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
        """Performs dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`oneflow.Tensor.to`, except optional
        arguments like `non_blocking` and `copy` should be passed as kwargs,
        not args, or they will not apply to the index tensors.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`oneflow.dtype`
            and :class:`oneflow.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            kwargs = {
                k: v
                for k, v in filter(
                    lambda t: t[0] != "device" and t[0] != "dtype", kwargs.items()
                )
            }
            sorted_indices = bind(
                self.sorted_indices, lambda t: t.to(data.device, **kwargs)
            )
            unsorted_indices = bind(
                self.unsorted_indices, lambda t: t.to(data.device, **kwargs)
            )
            return PackedSequence(
                data, self.batch_sizes, sorted_indices, unsorted_indices
            )

    @property
    def is_cuda(self):
        r"""Returns true if `self.data` stored on a gpu"""
        return self.data.is_cuda

    def is_pinned(self):
        r"""Returns true if `self.data` stored on in pinned memory"""
        return self.data.is_pinned()


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.utils.rnn.pack_padded_sequence.html.
    
    Packs a Tensor containing padded sequences of variable length.

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
    lengths = flow.as_tensor(lengths, dtype=flow.int64)
    assert (
        enforce_sorted == True
    ), "Only support enforce_sorted == True for now. Plesase Sort the input by length in a decreasing order."
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = flow.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)
    data, batch_sizes = flow._C.pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices, None)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.utils.rnn.pad_packed_sequence.html.
    
    Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *``, where `T` is the length
    of the longest sequence and `B` is the batch size. If ``batch_first`` is True,
    the data will be transposed into ``B x T x *`` format.

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~oneflow.nn.Module` wrapped in :class:`~oneflow.nn.DataParallel`.

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

    For example:

    .. code-block:: python

        >>> from oneflow.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> import oneflow as flow

        >>> seq = flow.tensor([[4,5,6], [1,2,0], [3,0,0]])
        >>> lens = [3, 2, 1]
        >>> packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=True)
        >>> packed.data
        tensor([4, 1, 3, 5, 2, 6], dtype=oneflow.int64)
        >>> packed.batch_sizes
        tensor([3, 2, 1], dtype=oneflow.int64)
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[4, 5, 6],
                [1, 2, 0],
                [3, 0, 0]], dtype=oneflow.int64)
        >>> lens_unpacked
        tensor([3., 2., 1.], dtype=oneflow.float32)


    """
    max_seq_length = sequence.batch_sizes.shape[0]
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError(
                "Expected total_length to be at least the length "
                "of the longest sequence in input, but got "
                "total_length={} and max sequence length being {}".format(
                    total_length, max_seq_length
                )
            )
    else:
        total_length = max_seq_length

    batch_sizes_t = sequence.batch_sizes.contiguous()
    assert (
        len(batch_sizes_t.shape) == 1
        and batch_sizes_t.device.type == "cpu"
        and batch_sizes_t.dtype == flow.int64
    ), f"'sequence.batch_sizes' should be a 1D CPU int64 tensor, but got {len(batch_sizes_t.shape)} D {batch_sizes_t.device.type} {batch_sizes_t.dtype} tensor"

    batch_sizes = batch_sizes_t.numpy()
    max_batch_size = int(batch_sizes[0])
    max_real_seq_length = batch_sizes_t.shape[0]
    max_seq_length = max_real_seq_length
    if total_length > 0:
        assert (
            total_length >= max_seq_length
        ), f"Expected total_length to be at least the length of the longest sequence in input, but got total_length={total_length} and max sequence length being {max_seq_length}"
        max_seq_length = total_length

    output_size = []  # == [max_seq_length, max_batch_size, *sequence.data.size()[1:]]
    output_size.append(max_seq_length)
    output_size.append(max_batch_size)
    output_size = output_size + list(sequence.data.shape[1:])
    padded_output = flow.full(
        output_size,
        padding_value,
        dtype=sequence.data.dtype,
        device=sequence.data.device,
        requires_grad=sequence.data.requires_grad,
    )
    # `padded_output` is leaf tensor which needs to be transformed into non-leaf tensor
    # when it requires grad by calling the `clone` method before the following
    # in-place operation to avoid runtime check error .
    if padded_output.requires_grad == True:
        padded_output = padded_output.clone()

    # This will be modified at every iteration, but we reserve memory for it now.
    tmp_view_size = output_size  # == [-1, -1, *sequence.data.size()[1:]]
    lengths = flow.empty(max_batch_size)
    data_offset = 0
    prev_batch_size = max_batch_size
    prev_i = 0
    lengths_idx = max_batch_size - 1
    for i in range(max_real_seq_length + 1):
        batch_size = batch_sizes[i] if i != max_real_seq_length else 0
        if batch_size != prev_batch_size:
            l = prev_batch_size * (i - prev_i)
            tmp_view_size[0] = i - prev_i
            tmp_view_size[1] = prev_batch_size
            padded_output[prev_i:i, 0:prev_batch_size] = sequence.data[
                data_offset : data_offset + l
            ].view(tmp_view_size)
            data_offset += l
            prev_i = i

        dec = prev_batch_size - batch_size
        if dec > 0:
            for j in range(dec):
                lengths[lengths_idx] = i
                lengths_idx = lengths_idx - 1
        prev_batch_size = batch_size

    if batch_first:
        permute_dims = [1, 0]
        for i in range(2, padded_output.ndim):
            permute_dims.append(i)
        padded_output = padded_output.permute(permute_dims)

    unsorted_indices = sequence.unsorted_indices
    if unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return (
            padded_output.index_select(batch_dim, unsorted_indices),
            lengths[unsorted_indices],
        )
    return padded_output, lengths


def pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.utils.rnn.pad_sequence.html.
    
    Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise. Default: False.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise

    For example:

    .. code-block:: python
    
        >>> from oneflow.nn.utils.rnn import pad_sequence
        >>> import oneflow as flow

        >>> a = flow.ones(25, 300)
        >>> b = flow.ones(22, 300)
        >>> c = flow.ones(15, 300)
        >>> out = pad_sequence([a, b, c])
        >>> out.size()
        oneflow.Size([25, 3, 300])

    """
    if isinstance(sequences, Tensor):
        sequences = sequences.unbind(0)

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    sequences_size = len(sequences)
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    lens = [seq.shape[0] for seq in sequences]
    lens.sort(reverse=True)
    max_len = lens[0]
    out_dims = [sequences_size, max_len] if batch_first else [max_len, sequences_size]
    out_dims = out_dims + list(trailing_dims)

    out = flow.full(
        out_dims,
        padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
        requires_grad=sequences[0].requires_grad,
    )
    for i in range(sequences_size):
        currseq = sequences[i]
        length_i = currseq.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out[i, 0:length_i] = currseq
        else:
            out[0:length_i, i] = currseq
    return out


def unpad_sequence(
    padded_sequences: Tensor, lengths: Tensor, batch_first: bool = False,
) -> List[Tensor]:
    """
    Unpad padded Tensor into a list of variable length Tensors

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: False.

    Returns:
        a list of :class:`Tensor` objects

    For example:

    .. code-block:: python

        >>> from oneflow.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> import oneflow as flow
        >>> import numpy as np

        >>> a = flow.ones(25, 300)
        >>> b = flow.ones(22, 300)
        >>> c = flow.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = flow.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> np.allclose(sequences[0].numpy(), unpadded_sequences[0].numpy())
        True
        >>> np.allclose(sequences[1].numpy(), unpadded_sequences[1].numpy())
        True
        >>> np.allclose(sequences[2].numpy(), unpadded_sequences[2].numpy())
        True
    """
    unpadded_sequences = []

    if not batch_first:
        padded_sequences = padded_sequences.permute((1, 0, 2))

    max_length = padded_sequences.shape[1]
    idx = flow.arange(max_length)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def pack_sequence(
    sequences: List[Tensor], enforce_sorted: bool = True
) -> PackedSequence:
    """Packs a list of variable length Tensors

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including zero.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.

    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object

    For example:

    .. code-block:: python
    
        >>> from oneflow.nn.utils.rnn import pack_sequence
        >>> import oneflow as flow

        >>> a = flow.tensor([1,2,3])
        >>> b = flow.tensor([4,5])
        >>> c = flow.tensor([6])
        >>> packed = pack_sequence([a, b, c])
        >>> packed.data
        tensor([1, 4, 6, 2, 5, 3], dtype=oneflow.int64)
        >>> packed.batch_sizes
        tensor([3, 2, 1], dtype=oneflow.int64)

    """
    lengths = flow.as_tensor([v.size(0) for v in sequences])
    return pack_padded_sequence(
        pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted
    )


def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]:
    """Unpacks PackedSequence into a list of variable length Tensors

    ``packed_sequences`` should be a PackedSequence object.

    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects

    For example:

    .. code-block:: python

        >>> from oneflow.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> import oneflow as flow

        >>> a = flow.tensor([1,2,3])
        >>> b = flow.tensor([4,5])
        >>> c = flow.tensor([6])
        >>> sequences = [a, b, c]
        >>> packed_sequences = pack_sequence(sequences)
        >>> packed_sequences.data
        tensor([1, 4, 6, 2, 5, 3], dtype=oneflow.int64)
        >>> packed_sequences.batch_sizes
        tensor([3, 2, 1], dtype=oneflow.int64)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        >>> unpacked_sequences
        [tensor([1, 2, 3], dtype=oneflow.int64), tensor([4, 5], dtype=oneflow.int64), tensor([6], dtype=oneflow.int64)]

    """

    padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first=True)
    unpacked_sequences = unpad_sequence(padded_sequences, lengths, batch_first=True)
    return unpacked_sequences


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
