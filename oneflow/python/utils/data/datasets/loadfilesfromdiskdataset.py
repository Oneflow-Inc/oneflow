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
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.common import get_file_binaries_from_pathnames

from typing import Iterable, Iterator


class LoadFilesFromDiskIterableDataset(IterableDataset):
    r""" :class:`LoadFilesFromDiskIterableDataset`.

    IterableDataset to load file binary streams from given pathnames,
    yield pathname and binary stream in a tuple.
    args:
        dataset: Iterable dataset that provides pathnames
        length: a nominal length of the dataset
    """

    def __init__(self, dataset: Iterable, length: int = -1):
        super().__init__()
        self.dataset: Iterable = dataset
        self.length: int = length

    def __iter__(self) -> Iterator[tuple]:
        yield from get_file_binaries_from_pathnames(self.dataset)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
