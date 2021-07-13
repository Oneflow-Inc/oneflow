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
from oneflow.python.utils.data.datapipes.utils.decoder import (
    Decoder,
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler,
)

from typing import Iterable, Iterator, Union, List, Tuple, Any, Callable
from io import BufferedIOBase


class RoutedDecoderIterDataPipe(IterDataPipe):
    r""" :class:`RoutedDecoderIterDataPipe`.

    Iterable datapipe to decode binary streams from input iterables,
    yield pathname and decoded binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: user defined decoder handlers, if None, basic and image decoder handlers will be set as default
        length: a nominal length of the datapipe
    """

    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        *,
        handlers: Union[None, List[Callable]] = None,
        length: int = -1
    ):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        if handlers:
            self.decoder = Decoder(handlers)
        else:
            self.decoder = Decoder(
                [decoder_basichandlers, decoder_imagehandler("torch")]
            )
        self.length: int = length

    def add_handler(self, handler: Callable) -> None:
        self.decoder.add_handler(handler)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
