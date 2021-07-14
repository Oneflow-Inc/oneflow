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
from oneflow.python.utils.data.dataset import IterableDataset
from oneflow.python.utils.data.datasets.common import get_file_pathnames_from_root

from typing import List, Union, Iterator


class ListDirFilesIterableDataset(IterableDataset):
    r""" :class:`ListDirFilesIterableDataset`

    IterableDataset to load file pathname(s) (path + filename), yield pathname from given disk root dir.
    args:
        root : root dir
        mask : a unix style filter string or string list for filtering file name(s)
        abspath : whether to return relative pathname or absolute pathname
        length : a nominal length of the dataset
    """

    def __init__(
        self,
        root: str = ".",
        masks: Union[str, List[str]] = "*.tar",
        *,
        abspath: bool = False,
        length: int = -1
    ):
        super().__init__()
        self.root: str = root
        self.masks: Union[str, List[str]] = masks
        self.abspath: bool = abspath
        self.length: int = length

    def __iter__(self) -> Iterator[str]:
        yield from get_file_pathnames_from_root(self.root, self.masks, self.abspath)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
