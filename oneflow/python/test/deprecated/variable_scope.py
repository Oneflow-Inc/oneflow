import numpy as np
import oneflow as of


@flow.global_function
def variable_scope_test_job_1(a=of.FixedTensorDef((1, 3, 6, 6))):
    with of.deprecated.variable_scope("job1_scope1"):
        convw = of.get_variable(
            "conv_weight",
            shape=(5, 3, 3, 3),
            dtype=a.dtype,
            initializer=of.random_uniform_initializer(),
            trainable=True,
        )
        conv = of.nn.conv2d(a, convw, 1, "SAME", "NCHW", name="conv")

        with of.deprecated.variable_scope("job1_scope2"):
            fcw = of.get_variable(
                "fc_weight",
                shape=(180, 10),
                dtype=a.dtype,
                initializer=of.random_uniform_initializer(),
                trainable=True,
            )
            fc = of.matmul(of.reshape(conv, (conv.shape[0], -1)), fcw, name="fc")
            fcb = of.get_variable(
                "fc_bias",
                shape=(10,),
                dtype=a.dtype,
                initializer=of.constant_initializer(1.0),
                trainable=True,
            )
            fc_bias = of.nn.bias_add(fc, fcb)

        fcw2 = of.get_variable(
            "fc2_weight",
            shape=(10, 20),
            dtype=a.dtype,
            initializer=of.random_uniform_initializer(),
            trainable=True,
        )
        fc2 = of.matmul(fc_bias, fcw2, name="fc2")

    print("conv_weight op name: ", convw.op_name)
    print("conv op name: ", conv.op_name)
    print("fc_weight op name: ", fcw.op_name)
    print("fc_bias op name: ", fcb.op_name)
    print("fc op name: ", fc.op_name)
    print("fc2_weight op name: ", fcw2.op_name)
    print("fc2 op name: ", fc2.op_name)

    return fc2


@flow.global_function
def variable_scope_test_job_2(a=of.FixedTensorDef((2, 5))):
    with of.deprecated.variable_scope("job2_scope1"):
        indices = of.get_variable(
            "gather_inds",
            shape=(2,),
            dtype=of.int32,
            initializer=of.constant_initializer(1),
            trainable=False,
        )
        output = of.gather(a, indices, axis=1)

    print("indices op name: ", indices.op_name)
    print("gather op name: ", output.op_name)
    return output


a1 = np.random.rand(1, 3, 6, 6).astype(np.float32)
a2 = np.arange(10, dtype=np.float32).reshape(2, 5)
ret1 = variable_scope_test_job_1.run(a1).get()
ret2 = variable_scope_test_job_2(a2).get()

print("Job1 result: ")
print(ret1)
print("shape: ", ret1.shape)
print("\n")
print("Job2 result: ")
print(ret2)
print("shape: ", ret2.shape)
