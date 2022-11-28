import oneflow
from oneflow.framework.docstr.utils import reset_docstr

reset_docstr(
    oneflow.add,
    r"""add(input, other)
    
    计算 `input` 和 `other` 的和。支持 element-wise、标量和广播形式的加法。
    公式为：

    .. math::
        out = input + other

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise 加法
        >>> x = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # 标量加法
        >>> x = 5
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # 广播加法
        >>> x = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """,
)
