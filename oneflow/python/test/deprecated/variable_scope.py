import oneflow as of
import numpy as np

config = of.ConfigProtoBuilder()
config.gpu_device_num(1)
of.init(config)


def variable_scope_test_job_1(a=of.input_blob_def((1, 3, 6, 6))):
    job_conf = of.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(of.float)
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
            fc = of.matmul(
                of.reshape(conv, (conv.static_shape[0], -1)), fcw, name="fc"
            )
            fcb = of.get_variable(
                "fc_bias",
                shape=(10,),
                dtype=a.dtype,
                initializer=of.constant_initializer(1.0),
                trainable=True,
            )
            fc_bias = of.nn.bias_add(fc, fcb)

    return fc_bias


def variable_scope_test_job_2(a=of.input_blob_def((2, 5))):
    job_conf = of.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(of.float)
    with of.deprecated.variable_scope("job2_scope1"):
        indices = of.get_variable(
            "gather_inds",
            shape=(2,),
            dtype=of.int32,
            initializer=of.constant_initializer(1),
            trainable=False,
        )
        output = of.gather(a, indices, axis=1)
        return output


of.add_job(variable_scope_test_job_1)
of.add_job(variable_scope_test_job_2)


with of.Session() as sess:
    a1 = np.random.rand(1, 3, 6, 6).astype(np.float32)
    a2 = np.arange(10, dtype=np.float32).reshape(2, 5)
    ret1 = sess.run(variable_scope_test_job_1, a1).get()
    ret2 = sess.run(variable_scope_test_job_2, a2).get()

    print("Job1 result: ")
    print(ret1)
    print("shape: ", ret1.shape)
    print("\n")
    print("Job2 result: ")
    print(ret2)
    print("shape: ", ret2.shape)
