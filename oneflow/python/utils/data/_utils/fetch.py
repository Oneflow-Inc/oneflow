r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import time
import numpy as np
import oneflow as flow

class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
    
    def _get_slice_obj(self, key, shape):
        def get_or_default(x, default):
            return x if x is not None else default

        def get_canonical_index(index, length, *, start=0):
            if index < 0:
                index += length
            return max(min(index, length), start)

        def get_slice_if_int(x):
            if isinstance(x, slice):
                return x
            return slice(x, x + 1)

        if isinstance(key, tuple):
            assert all(isinstance(x, (slice, int)) for x in key)
        else:
            assert isinstance(key, (slice, int))
            key = (key,)

        key = list(map(get_slice_if_int, key))

        assert len(key) <= len(shape)
        for i in range(len(key), len(shape)):
            key += (slice(None, None, None),)

        starts = [
            get_canonical_index(get_or_default(x.start, 0), shape[i])
            for i, x in enumerate(key)
        ]
        stops = [
            get_canonical_index(
                get_or_default(x.stop, shape[i]), shape[i], start=starts[i]
            )
            for i, x in enumerate(key)
        ]
        steps = [get_or_default(x.step, 1) for x in key]
        assert all(x > 0 for x in steps)
        # np.abs is for compatibility of negative steps in the future
        shapes = (np.abs(np.array(stops) - np.array(starts)) - 1) // np.abs(
            np.array(steps)
        ) + 1
        shapes = shapes.tolist()
        return starts, stops, steps, shapes

    def fetch(self, possibly_batched_index):
        t1 = time.time()
        # self.dataset[1024] cost 0.0023-0.0025
        if self.auto_collation:
            # torch one slice cost >>>>> 0.00011 s
            # flow  one slice cost >>>>> 0.00016-0.00022 s
            data = [self.dataset[idx] for idx in possibly_batched_index]
            # data = []
            # for idx in possibly_batched_index:
            #     start, stop, step, _ = self._get_slice_obj(idx, self.dataset[idx].shape)
            #     res = flow.experimental.slice(self, list(zip(start, stop, step)))
            #     data.append(res.squeeze(dim=[0]))
        else:
            data = self.dataset[possibly_batched_index]
        t2 = time.time()
        res = self.collate_fn(data)
        t3 = time.time()
        # torch dataset[idx] cost: 0.015226125717163086 ; self.collate_fn(data) cost: 0.00034356117248535156
        # flow  dataset[idx] cost: 0.5628399848937988 ;   self.collate_fn(data) cost: 0.01665043830871582
        print("fetch.py iter >>> dataset[idx] cost:", t2-t1, "; self.collate_fn(data) cost:", t3-t2)
        return res
