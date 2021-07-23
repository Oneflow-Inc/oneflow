import os
import sys
import traceback
import warnings
from typing import Any, Callable, Generic, List, Optional, Sequence, TypeVar

import oneflow as flow


class ExceptionWrapper(object):
    """Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        """Reraises the wrapped exception in the current thread"""
        msg = "Caught {} {}.\nOriginal {}".format(
            self.exc_type.__name__, self.where, self.exc_msg
        )
        if self.exc_type == KeyError:
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            raise self.exc_type(message=msg)
        raise self.exc_type(msg)


string_classes = (str, bytes)
from . import (
    BatchSampler,
    Dataset,
    IterableDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    _utils,
)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]
default_collate: _collate_fn_t = _utils.collate.default_collate


class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )
        else:
            return _utils.fetch._IterableDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )


class _InfiniteConstantSampler(Sampler):
    """Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~flow.utils.data.IterableDataset`.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None


class DataLoader(Generic[T_co]):
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~flow.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`flow.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers samples prefetched across all workers. (default: ``2``)
        persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in OneFlow.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~flow.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess OneFlow can make because OneFlow
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, OneFlow can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~flow.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.
    """

    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    drop_last: bool
    timeout: float
    sampler: Sampler
    prefetch_factor: int
    _iterator: Optional["_BaseDataLoaderIter"]
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False
    ):
        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; use num_workers=0 to disable multiprocessing."
            )
        if num_workers >= 1:
            warnings.warn(
                "Not support multiprocessing dataloader yet, we will temporary set num_workers=0!"
            )
            num_workers = 0
        if timeout < 0:
            raise ValueError("timeout option should be non-negative")
        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing.let num_workers > 0 to enable multiprocessing."
            )
        assert prefetch_factor > 0
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={}".format(
                        shuffle
                    )
                )
            elif sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={}".format(
                        sampler
                    )
                )
            elif batch_sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified batch_sampler option, but got batch_sampler={}".format(
                        batch_sampler
                    )
                )
        else:
            self._dataset_kind = _DatasetKind.Map
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching and is mutually exclusive with drop_last"
                )
        if sampler is None:
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            elif shuffle:
                sampler = RandomSampler(dataset, generator=generator)
            else:
                sampler = SequentialSampler(dataset)
        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator
        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert
        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers
        self.__initialized = True
        self._IterableDataset_len_called = None
        self._iterator = None

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0 or self.num_workers == 1:
            return _SingleProcessDataLoaderIter(self)
        else:
            raise NotImplementedError("Multiprocessing dataloader is not support yet!")

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                "{} attribute should not be set after {} is initialized".format(
                    attr, self.__class__.__name__
                )
            )
        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self) -> "_BaseDataLoaderIter":
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)
            if self.batch_size is not None:
                from math import ceil

                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = flow.Tensor([0], dtype=flow.int64).uniform_().numpy().item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(
            self.__class__.__name__
        )

    def __iter__(self) -> "_BaseDataLoaderIter":
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data()
        self._num_yielded += 1
        if (
            self._dataset_kind == _DatasetKind.Iterable
            and self._IterableDataset_len_called is not None
            and (self._num_yielded > self._IterableDataset_len_called)
        ):
            warn_msg = "Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} samples have been fetched. ".format(
                self._dataset, self._IterableDataset_len_called, self._num_yielded
            )
            if self._num_workers > 1:
                warn_msg += "Multiprocessing dataloader is not support yet!"
            warnings.warn(warn_msg)
        return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert 0 <= self._num_workers <= 1
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        index = self._next_index()
        return self._dataset_fetcher.fetch(index)
