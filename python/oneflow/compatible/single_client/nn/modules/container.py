from collections import OrderedDict
import collections.abc
from itertools import islice
import operator
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from typing import Any, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
T = TypeVar('T')
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)