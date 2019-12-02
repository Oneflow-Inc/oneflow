import oneflow as flow
import numpy as np

def test_basic(test_case):
  flow.config.gpu_device_num(1)
  flow.config.default_data_type(flow.float)

  @flow.function
  def ReluJob(x = flow.input_blob_def((10,))):
      return flow.keras.activations.relu(x)

  ratios = [-2, -1, 0, 1, 2]
  ones = np.ones((10,), dtype=np.float32)
  for r in ratios:
      x = ones * r
      of_ret = ReluJob(x).get()
      expected = x * (r > 0)
      test_case.assertTrue(np.allclose(expected, of_ret))
