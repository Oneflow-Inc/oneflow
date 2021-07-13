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
    validate_pathname_binary_tuple,
)
from typing import Iterable, Iterator, Tuple, Optional, IO, cast
from io import BufferedIOBase

import os
import tarfile
import warnings


class ReadFilesFromTarIterDataPipe(IterDataPipe):
    r""" :class:`ReadFilesFromTarIDP`.

    Iterable datapipe to extract tar binary streams from input iterable which contains tuples of
    pathname and tar binary stream, yields pathname and extracted binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathname and tar binary stream in tuples
        length: a nominal length of the datapipe
    """

    def __init__(
        self, datapipe: Iterable[Tuple[str, BufferedIOBase]], length: int = -1
    ):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        if not isinstance(self.datapipe, Iterable):
            raise TypeError(
                "datapipe must be Iterable type but got {}".format(type(self.datapipe))
            )
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(
                    fileobj=cast(Optional[IO[bytes]], data_stream), mode="r:*"
                )
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(
                            "failed to extract file {} from source tarfile {}".format(
                                tarinfo.name, pathname
                            )
                        )
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(
                        os.path.join(pathname, tarinfo.name)
                    )
                    # Add a reference of the source tarfile into extracted_fobj, so the source
                    # tarfile handle won't be released until all the extracted file objs are destroyed.
                    # Add `# type: ignore` to silence mypy's type checker
                    extracted_fobj.source_tarfile_ref = tar  # type: ignore
                    # typing.cast is used here to silence mypy's type checker
                    yield (inner_pathname, cast(BufferedIOBase, extracted_fobj))
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted tarfile stream {} due to: {}, abort!".format(
                        pathname, e
                    )
                )
                raise e

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
