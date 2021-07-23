import builtins
from typing import Iterator, Union, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np
import oneflow as flow
from oneflow.framework.tensor import Tensor
T_co = TypeVar('T_co', covariant=True)

class Sampler(Generic[T_co]):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~flow.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~flow.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

class RandomSampler(Sampler[int]):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool=False, num_samples: Optional[int]=None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        if not isinstance(self.replacement, bool):
            raise TypeError('replacement should be a boolean value, but got replacement={}'.format(self.replacement))
        if self._num_samples is not None and (not replacement):
            raise ValueError('With replacement=False, num_samples should not be specified, since a random permute will be performed.')
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError('num_samples should be a positive integer value, but got num_samples={}'.format(self.num_samples))

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = flow.Generator()
            generator.manual_seed(int(flow.Tensor(1, dtype=flow.int64).xavier_uniform_().numpy()[0]))
        else:
            generator = self.generator
        if self.replacement:
            raise NotImplementedError('Not support replacement yet!')
        else:
            yield from np.random.permutation(n).tolist()

    def __len__(self):
        return self.num_samples

class SubsetRandomSampler(Sampler[int]):
    """Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in flow.randperm(len(self.indices), generator=self.generator))

    def __len__(self):
        return len(self.indices)

class BatchSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, but got batch_size={}'.format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError('drop_last should be a boolean value, but got drop_last={}'.format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and (not self.drop_last):
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size