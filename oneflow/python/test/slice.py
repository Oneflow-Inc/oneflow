import oneflow as of
import numpy as np
from termcolor import colored


def slice_into_two_part():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.FixedTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        part1 = of.slice(input_blob, (None, 0, 0), (None, 2, None))
        part2 = of.slice(input_blob, (None, 2, 0), (None, 3, None))
        return part1, part2

    input = np.random.rand(2, 5, 4).astype(np.float32)
    p1, p2 = slice(input).get()

    ref1 = input[:, :2, :]
    ref2 = input[:, 2:5, :]
    print("-" * 80)
    print(colored("slice_into_two_part test", "yellow"))
    print("input shape:", input.shape)
    print("part1 shape:", ref1.shape)
    print("part2 shape:", ref2.shape)
    assert np.allclose(p1.ndarray(), ref1)
    assert np.allclose(p2.ndarray(), ref2)
    print(colored("slice_into_two_part test passed", "green"))


def slice_at_two_dim():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.FixedTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice(input_blob, (None, 0, 2), (None, 2, None))

    input = np.random.rand(2, 5, 4).astype(np.float32)
    ret = slice(input).get().ndarray()
    ref = input[:, 0:2, 2:]

    print("-" * 80)
    print(colored("slice_at_two_dim test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("slice_at_two_dim test passed", "green"))


def dynamic_slice():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice(input_blob, (None, 1, None), (None, None, None))

    input = np.random.rand(2, 4, 4).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, 1:, :]

    print("-" * 80)
    print(colored("dynamic_slice test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("dynamic_slice test passed", "green"))


def dynamic_slice_at_two_dim():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 2, 3), dtype=of.float32)):
        return of.slice(input_blob, (None, 2, None, 1), (None, None, None, None))

    input = np.random.rand(2, 3, 2, 2).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, 2:, :, 1:]

    print("-" * 80)
    print(colored("dynamic_slice_at_two_dim test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("dynamic_slice_at_two_dim test passed", "green"))


def slice_with_stride2():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice_v3(input_blob, [(None, None, None), (1, None, 2)])

    input = np.random.rand(2, 5, 4).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, 1::2, :]

    print("-" * 80)
    print(colored("slice_with_stride2 test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("slice_with_stride2 test passed", "green"))


def slice_at_two_dim_with_stride():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice_v3(input_blob, [(None, None, None), (1, None, 3), (None, None, 2)])

    input = np.random.rand(2, 5, 4).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, 1::3, ::2]

    print("-" * 80)
    print(colored("slice_at_two_dim_with_stride test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("slice_at_two_dim_with_stride test passed", "green"))


def slice_with_neg_index():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice_v3(input_blob, [(None, None, None), (2, -2, 1)])

    input = np.random.rand(2, 5, 4).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, 2:-2, :]

    print("-" * 80)
    print(colored("slice_with_neg_index test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("slice_with_neg_index test passed", "green"))


def slice_with_neg_stride():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice_v3(input_blob, [(None, None, None), (3, -4, -1)])

    input = np.random.rand(2, 5, 4).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, 3:-4:-1, :]

    print("-" * 80)
    print(colored("slice_with_neg_stride test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("slice_with_neg_stride test passed", "green"))


def slice_with_neg_stride2():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice(input_blob=of.MirroredTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        return of.slice_v3(input_blob, [(None, None, None), (-1, 1, -2)])

    input = np.random.rand(2, 5, 4).astype(np.float32)
    ret = slice([input]).get().ndarray_list()[0]
    ref = input[:, -1:1:-2, :]

    print("-" * 80)
    print(colored("slice_with_neg_stride2 test", "yellow"))
    print("input shape:", input.shape)
    print("of result shape:", ret.shape)
    print("np result shape:", ref.shape)
    assert np.allclose(ret, ref)
    print(colored("slice_with_neg_stride2 test passed", "green"))


def slice_grad():
    of.clear_default_session()
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)
    func_config.train.primary_lr(0.01)
    func_config.train.model_update_conf(dict(naive_conf={}))

    grad = None

    def get_slice_grad(blob):
        global grad
        grad = blob.ndarray()

    @of.function(func_config)
    def slice(input_blob=of.FixedTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        x = of.get_variable(
            shape=(2, 5, 4),
            dtype=of.float32,
            initializer=of.random_uniform_initializer(2),
            name="variable",
        )
        of.assign(x, input_blob)
        x = of.identity(x)
        of.watch_diff(x, get_slice_grad)
        y = of.slice_v3(x, [(None, None, None), (2, -2, 1)])
        of.losses.add_loss(y)
        return y

    input = np.random.rand(2, 5, 4).astype(np.float32)
    slice(input).get()
    ref = np.zeros(input.shape, dtype=np.float32)
    ref[:, 2:-2, :] = np.ones(input[:, 2:-2, :].shape, dtype=np.float32)

    print("-" * 80)
    print(colored("slice_grad test", "yellow"))
    print("input shape:", input.shape)
    print("of grad shape:", grad.shape)
    print("np grad shape:", ref.shape)
    assert np.allclose(grad, ref)
    print(colored("slice_grad test passed", "green"))


if __name__ == "__main__":
    slice_into_two_part()
    slice_at_two_dim()
    dynamic_slice()
    dynamic_slice_at_two_dim()
    slice_with_stride2()
    slice_at_two_dim_with_stride()
    slice_with_neg_index()
    slice_with_neg_stride()
    slice_with_neg_stride2()
    slice_grad()
