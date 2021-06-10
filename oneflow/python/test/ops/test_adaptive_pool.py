

import oneflow as flow
import numpy as np
import oneflow.typing as tp


@flow.global_function()
def adaptive_avgpool2d_Job(x: tp.Numpy.Placeholder((1, 2, 6, 6))
) -> tp.Numpy:
    pool_out = flow.nn.adaptive_avg_pool2d(
        input=x,
        output_size=(2, 2)
    )

    return pool_out


x = np.random.randn(1, 2, 6, 6).astype(np.float32)
out = adaptive_avgpool2d_Job(x)

