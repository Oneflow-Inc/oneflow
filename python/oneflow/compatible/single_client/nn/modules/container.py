import collections.abc
import operator
from collections import OrderedDict
from itertools import islice
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module

T = TypeVar("T")
if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
