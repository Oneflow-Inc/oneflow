import oneflow
from oneflow.framework.docstr.utils import reset_docstr

reset_docstr(
    oneflow.nn.ReLU,
    r"""ReLU(inplace=False)
    
    ReLU 激活函数，对张量中的每一个元素做 element-wise 运算，公式如下:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    参数:
        inplace: 是否做 in-place 操作。 默认为 ``False``

    形状:
        - Input: :math:`(N, *)` 其中 `*` 的意思是，可以指定任意维度
        - Output: :math:`(N, *)` 输入形状与输出形状一致

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> relu = flow.nn.ReLU()
        >>> ndarr = np.asarray([1, -2, 3])
        >>> x = flow.Tensor(ndarr)
        >>> relu(x)
        tensor([1., 0., 3.], dtype=oneflow.float32)

    """,
)
