import oneflow as flow
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.enable_eager_execution()


def test_scatter_nd_update(test_case):
    # manufacture inputs
    ref = np.random.randint(1024, size=(10,)).astype(np.float32)
    indices = np.arange(10)
    np.random.shuffle(indices)
    indices = indices[:5].reshape(5, 1).astype(np.int32)
    updates = np.random.randint(1024, size=(5)).astype(np.float32)

    # tf scatter_nd_update output
    tf_out = tf.compat.v1.scatter_nd_update(
        tf.Variable(ref), tf.Variable(indices), tf.Variable(updates)
    ).numpy()

    # oneflow scatter_nd_update output
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def scatter_nd_update_fn(
        input_def=flow.MirroredTensorDef(ref.shape, dtype=flow.float),
        indices_def=flow.MirroredTensorDef(indices.shape, dtype=flow.int32),
        updates_def=flow.MirroredTensorDef(updates.shape, dtype=flow.float),
    ):
        return flow.scatter_nd_update(input_def, indices_def, updates_def)

    of_out = scatter_nd_update_fn([ref], [indices], [updates]).get().ndarray_list()[0]

    # compare
    # print("tf_out", tf_out)
    # print("of_out", of_out)
    test_case.assertTrue(np.allclose(tf_out, of_out))
