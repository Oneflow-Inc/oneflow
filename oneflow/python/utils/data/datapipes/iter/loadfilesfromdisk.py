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
from oneflow.python.utils.data import IterDataPipe
from oneflow.python.utils.data.datapipes.utils.common import (
    get_file_binaries_from_pathnames,
)
from typing import Iterable, Iterator, Tuple
from io import BufferedIOBase


class LoadFilesFromDiskIterDataPipe(IterDataPipe):
    r""" :class:`LoadFilesFromDiskIterDataPipe`.

    Iterable Datapipe to load file binary streams from given pathnames,
    yield pathname and binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathnames
        length: a nominal length of the datapipe
    """

    def __init__(self, datapipe: Iterable[str], length: int = -1):
        super().__init__()
        self.datapipe: Iterable = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        yield from get_file_binaries_from_pathnames(self.datapipe)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
