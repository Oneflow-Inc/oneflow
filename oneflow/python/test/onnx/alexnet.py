import numpy as np
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import onnxruntime as ort

_MODEL_LOAD = "/home/dev/files/of_model/"

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):
  weight_shape = (filters, input.shape[1], kernel_size, kernel_size)
  weight = flow.get_variable(
    name + "-weight",
    shape=weight_shape,
    dtype=input.dtype,
    initializer=weight_initializer,
  )
  output = flow.nn.conv2d(
    input, weight, strides, padding, data_format, dilation_rate, name=name
  )
  if use_bias:
    bias = flow.get_variable(
      name + "-bias",
      shape=(filters,),
      dtype=input.dtype,
      initializer=bias_initializer,
    )
    output = flow.nn.bias_add(output, bias, data_format)

  if activation is not None:
    if activation == op_conf_util.kRelu:
      output = flow.nn.relu(output)
    else:
      raise NotImplementedError

  return output


def alexnet(images, labels, trainable=True):
  transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
  conv1 = _conv2d_layer("conv1", transposed, filters=64, kernel_size=11, strides=4, padding="VALID")

  pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

  conv2 = _conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

  pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

  conv3 = _conv2d_layer("conv3", pool2, filters=384)

  conv4 = _conv2d_layer("conv4", conv3, filters=384)

  conv5 = _conv2d_layer("conv5", conv4, filters=256)

  pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

  def _get_initializer():
    kernel_initializer = op_conf_util.InitializerConf()
    kernel_initializer.truncated_normal_conf.std = 0.816496580927726
    return kernel_initializer

  if len(pool5.shape) > 2:
    pool5 = flow.reshape(pool5, shape=(pool5.shape[0], -1))

  fc1 = flow.layers.dense(
    inputs=pool5,
    units=4096,
    activation=flow.keras.activations.relu,
    use_bias=False,
    kernel_initializer=_get_initializer(),
    bias_initializer=False,
    trainable=trainable,
    name="fc1",
  )

  dropout1 = fc1

  fc2 = flow.layers.dense(
    inputs=dropout1,
    units=4096,
    activation=flow.keras.activations.relu,
    use_bias=False,
    kernel_initializer=_get_initializer(),
    bias_initializer=False,
    trainable=trainable,
    name="fc2",
  )

  dropout2 = fc2

  fc3 = flow.layers.dense(
    inputs=dropout2,
    units=1001,
    activation=None,
    use_bias=False,
    kernel_initializer=_get_initializer(),
    bias_initializer=False,
    trainable=trainable,
    name="fc3",
  )

  return fc3

def main():
  func_config = flow.FunctionConfig()
  func_config.default_data_type(flow.float)
  @flow.function(func_config)
  def alexnet_eval_job(x=flow.FixedTensorDef((1,227,227,3))):
    with flow.distribute.consistent_strategy():
      return alexnet(x, None, False)

  check_point = flow.train.CheckPoint()
  check_point.load(_MODEL_LOAD)

  onnx_proto = flow.onnx.export(alexnet_eval_job, _MODEL_LOAD)

  ipt = np.random.uniform(low=-10, high=10,
                        size=(1,227,227,3)).astype(np.float32)

  sess = ort.InferenceSession(onnx_proto.SerializeToString())
  onnx_res = sess.run([], {'Input_1/out': ipt})
  oneflow_res = alexnet_eval_job(ipt).get().ndarray()
  print(onnx_res)
  print(oneflow_res)
  assert np.allclose(onnx_res, oneflow_res, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
  main()

