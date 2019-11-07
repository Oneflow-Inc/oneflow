import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()
flow.config.data_port(10000)
flow.config.ctrl_port(20000)

flow.config.default_data_type(flow.float)

def Foo():
  flow.clear_default_session()
  @flow.function
  def IdJob(x = flow.input_blob_def((2, 5))):
      return x

  index = [-2, -1, 0, 1, 2]
  data = []
  for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
  for x in data:  print(IdJob(x).get())

Foo()
Foo()
