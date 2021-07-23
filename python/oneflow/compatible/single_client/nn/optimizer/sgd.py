import collections
from typing import Callable, Dict, Iterator, List, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.parameter import Parameter

from .optimizer import Optimizer, ParamGroup
