from absl.testing import absltest
import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

data = np.ones((10,), dtype=np.float32)

class TestFunction(absltest.TestCase):
    def tearDown(self):
        flow.clear_default_session()

    def test_none(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))): return None
        self.assertTrue(ReluJob(data) == None)
        
    def test_ret_blob(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return flow.keras.activations.relu(x)
        self.assertTrue(np.allclose(ReluJob(data).get(), data))

    def test_ret_list(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return [flow.keras.activations.relu(x)]
        ret = ReluJob(data).get()
        self.assertTrue(isinstance(ret, list))
        self.assertTrue(np.allclose(ret[0], data))

    def test_ret_tuple(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return (flow.keras.activations.relu(x),)
        ret = ReluJob(data).get()
        self.assertTrue(isinstance(ret, tuple))
        self.assertTrue(np.allclose(ret[0], data))

    def test_ret_dict(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return dict(ret=flow.keras.activations.relu(x))
        ret = ReluJob(data).get()
        self.assertTrue('ret' in ret)
        self.assertTrue(np.allclose(ret['ret'], data))

    def test_ret_list_of_tuple(self):
        @flow.function
        def ReluJob(x = flow.input_blob_def((10,))):
            return [(flow.keras.activations.relu(x),)]
        ret = ReluJob(data).get()
        self.assertTrue(isinstance(ret, list))
        self.assertTrue(isinstance(ret[0], tuple))
        self.assertTrue(np.allclose(ret[0][0], data))

if __name__ == '__main__': absltest.main()
