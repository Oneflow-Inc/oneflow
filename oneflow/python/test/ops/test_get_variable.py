import numpy as np
import oneflow as flow


def test_get_variable_with_same_name(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def get_v():
        return flow.get_variable(
            name="var",
            shape=(5, 2),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(),
        )

    @flow.global_function(func_config)
    def TestJob0():
        v1 = get_v()
        v2 = get_v()
        return v1, v2

    @flow.global_function(func_config)
    def TestJob1():
        return get_v()

    check_point = flow.train.CheckPoint()
    check_point.init()
    j0_v1, j0_v2 = TestJob0().get()
    j1_v = TestJob1().get()
    test_case.assertTrue(np.array_equal(j0_v1.numpy(), j0_v2.numpy()))
    test_case.assertTrue(np.array_equal(j0_v1.numpy(), j1_v.numpy()))


def test_get_job_shared_variable(test_case):
    flow.clear_default_session()

    def get_var(name, shape=(2, 5), dtype=flow.float, trainable=False):
        return flow.get_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            initializer=flow.random_uniform_initializer(),
        )

    learning_rate = 1e-2
    train_func_config = flow.FunctionConfig()
    train_func_config.train.primary_lr(learning_rate)
    train_func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(train_func_config)
    def train(x_def=flow.FixedTensorDef(shape=(2, 5), dtype=flow.float)):
        var = get_var("var", trainable=True)
        loss = var + x_def
        flow.losses.add_loss(loss)
        return loss

    @flow.global_function()
    def eval():
        return get_var("var")

    check_point = flow.train.CheckPoint()
    check_point.init()

    variables = []
    for i in range(10):
        input = np.random.rand(2, 5).astype(np.single)
        train(input).get()
        var = eval().get().numpy()
        # print("variable at iter {}:".format(i), var)
        if i > 0:
            test_case.assertTrue(
                np.allclose(var, (variables[-1] - learning_rate / var.size))
            )

        variables.append(var)


def test_get_job_inter_and_intra_shared_variable(test_case):
    flow.clear_default_session()

    variable_shape = (2, 5)

    def get_var(name, shape=variable_shape, dtype=flow.float, trainable=False):
        return flow.get_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            initializer=flow.random_uniform_initializer(),
        )

    learning_rate = 1e-2
    train_func_config = flow.FunctionConfig()
    train_func_config.train.primary_lr(learning_rate)
    train_func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(train_func_config)
    def train(x_def=flow.FixedTensorDef(shape=variable_shape, dtype=flow.float)):
        var = get_var("var", trainable=True)
        loss = var + x_def
        flow.losses.add_loss(loss)
        return loss

    @flow.global_function()
    def eval():
        v1 = get_var("var")
        v2 = get_var("var")
        return v1, v2

    check_point = flow.train.CheckPoint()
    check_point.init()

    variables = []
    for i in range(10):
        input = np.random.rand(*variable_shape).astype(np.single)
        train(input).get()
        var1, var2 = eval().get()
        test_case.assertTrue(np.array_equal(var1.numpy(), var2.numpy()))
        # print("variable at iter {}:".format(i), var1.numpy())
        if i > 0:
            test_case.assertTrue(
                np.allclose(var1.numpy(), (variables[-1] - learning_rate / var1.size))
            )

        variables.append(var1.numpy())
