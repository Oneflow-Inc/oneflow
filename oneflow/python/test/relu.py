import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

class TestRelu(flow.unittest.TestCase):
    def testOnes(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return flow.keras.activations.relu(x)
        ones = np.ones((10,), dtype=np.float32)
        self.assertTrue(np.allclose(ReluJob(ones).get().ndarray(), ones))

    def testZeros(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return flow.keras.activations.relu(x)
        zeros = np.zeros((10,), dtype=np.float32)
        self.assertTrue(np.allclose(ReluJob(zeros).get().ndarray(), zeros))

    def testNegative(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return flow.keras.activations.relu(x)
        data = np.ones((10,), dtype=np.float32) * -1
        zeros = np.zeros((10,), dtype=np.float32)
        self.assertTrue(np.allclose(ReluJob(data).get().ndarray(), zeros))

flow.unittest.run_if_this_is_main()
