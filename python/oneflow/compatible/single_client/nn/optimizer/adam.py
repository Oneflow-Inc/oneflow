from typing import List, Dict, Callable, Union, Iterator, Tuple
import collections
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.parameter import Parameter
from oneflow.compatible.single_client.python.nn.optimizer.optimizer import Optimizer, ParamGroup