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
from oneflow.python.utils.data import IterDataPipe, functional_datapipe
from typing import Callable, TypeVar, Iterator, Optional, Tuple, Dict

from .callable import MapIterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("filter")
class FilterIterDataPipe(MapIterDataPipe[T_co]):
    r""" :class:`FilterIterDataPipe`.

    Iterable DataPipe to filter elements from datapipe according to filter_fn.
    args:
        datapipe: Iterable DataPipe being filterd
        filter_fn: Customized function mapping an element to a boolean.
        fn_args: Positional arguments for `filter_fn`
        fn_kwargs: Keyword arguments for `filter_fn`
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        filter_fn: Callable[..., bool],
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(datapipe, fn=filter_fn, fn_args=fn_args, fn_kwargs=fn_kwargs)

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            res = self.fn(data, *self.args, **self.kwargs)
            if not isinstance(res, bool):
                raise ValueError(
                    "Boolean output is required for "
                    "`filter_fn` of FilterIterDataPipe"
                )
            if res:
                yield data

    def __len__(self):
        raise (NotImplementedError)
