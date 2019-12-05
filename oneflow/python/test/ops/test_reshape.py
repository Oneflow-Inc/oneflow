import oneflow as flow
import numpy as np


def test_reshape(test_case):
    flow.config.gpu_device_num(2)
    flow.config.default_data_type(flow.float)

    @flow.function
    def ReshapeJob(x=flow.input_blob_def((10, 20, 20))):
        return flow.reshape(x, (200, 20))

    random_array = np.random.rand(10, 20, 20).astype(np.float32)
    test_case.assertTrue(
        np.allclose(ReshapeJob(random_array).get(), np.reshape(random_array, (200, 20)))
    )
