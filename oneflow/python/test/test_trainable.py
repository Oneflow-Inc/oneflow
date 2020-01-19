import oneflow as flow
import numpy as np

batch_size = 8
iterations = 100

def linear(input, units, name=None, trainable=True):
    assert name is not None

    in_shape = input.static_shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    inputs = (
        flow.reshape(input, (-1, in_shape[-1])) if in_num_axes > 2 else input
    )

    weight = flow.get_variable(
        name="{}-weight".format(name),
        shape=(units, inputs.static_shape[1]),
        dtype=inputs.dtype,
        initializer=flow.random_normal_initializer(stddev=0.02),
        trainable=trainable,
        model_name="weight",
    )

    out = flow.matmul(
        a=inputs,
        b=weight,
        transpose_b=True,
        name=name + "matmul",
    )

    bias = flow.get_variable(
        name="{}-bias".format(name),
        shape=(units,),
        dtype=inputs.dtype,
        initializer=flow.random_normal_initializer(),
        trainable=trainable,
        model_name="bias",
    )

    out = flow.nn.bias_add(
        out, bias, name=name + "_bias_add"
    )

    out = (
        flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out
    )
    return out

@flow.function
def test_trainable(input=flow.input_blob_def((batch_size, 100))):
    flow.config.train.primary_lr(0.0001)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    ret = linear(input, 1, trainable=False, name="linear0")
    loss = linear(input, 1, trainable=True, name="linear1") + ret
    flow.losses.add_loss(loss)
    return (ret, loss)

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float32)
check_point = flow.train.CheckPoint()
check_point.init()

# when the input is fixed, the loss should not drop but it does
z = np.random.normal(0, 1, size=(batch_size, 100)).astype(np.float32)
for itr in range(iterations):
    (ret, loss) = test_trainable(z).get()

    if  (itr + 1) % 10 == 0:
        print(itr + 1, "th batch:")
        print("ret:", ret.mean())
        print("loss:", loss.mean())
