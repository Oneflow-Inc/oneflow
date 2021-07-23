from typing import Optional

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.common_types import (
    _size_1_t,
    _size_2_t,
    _size_3_t,
)
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.modules.utils import (
    _pair,
    _single,
    _triple,
)
from oneflow.compatible.single_client.python.ops.nn_ops import (
    calc_pool_padding,
    get_dhw_offset,
)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
