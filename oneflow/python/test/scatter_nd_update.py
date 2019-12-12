import oneflow as of
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.enable_eager_execution()
# assert(tf.executing_eagerly())

g_of_scatter_nd_update_updates_diff = None
g_of_scatter_nd_update_input_diff = None


def of_scatter_nd_update(input, indices, updates):
    @of.function
    def scatter_nd_update_job(
        input_blob=of.input_blob_def(input.shape, dtype=of.float),
        indices_blob=of.input_blob_def(indices.shape, dtype=of.int32),
        updates_blob=of.input_blob_def(updates.shape, dtype=of.float),
    ):
        return of.local_scatter_nd_update(
            input_blob, indices_blob, updates_blob
        )

    return scatter_nd_update_job(input, indices, updates).get()


@of.function
def scatter_nd_update_train_job(
    indices_blob=of.input_blob_def((5, 1), dtype=of.int32)
):
    of.config.train.model_update_conf(dict(naive_conf={}))
    of.config.train.primary_lr(0.1)
    input = of.get_variable(
        "input",
        shape=(10,),
        dtype=of.float32,
        initializer=of.constant_initializer(0),
        distribute=of.distribute.split(axis=0),
    )
    updates = of.get_variable(
        "updates",
        shape=(5,),
        dtype=of.float32,
        initializer=of.constant_initializer(2),
        distribute=of.distribute.broadcast(),
    )

    input = input.with_split_distribute(axis=0)
    output = of.local_scatter_nd_update(input, indices_blob, updates)
    of.losses.add_loss(output)

    def input_diff(blob):
        global g_of_scatter_nd_update_input_diff
        g_of_scatter_nd_update_input_diff = blob.ndarray()
        print("scatter_nd_update input_diff: \n", blob)

    def updates_diff(blob):
        global g_of_scatter_nd_update_updates_diff
        g_of_scatter_nd_update_updates_diff = blob.ndarray()
        print("scatter_nd_update updates_diff: \n", blob)

    of.watch(
        input,
        lambda blob: print(
            "scatter_nd_update forward input: \n", blob.ndarray()
        ),
    )
    of.watch_diff(input, input_diff)
    of.watch(
        updates,
        lambda blob: print(
            "scatter_nd_update forward updates: \n", blob.ndarray()
        ),
    )
    of.watch_diff(updates, updates_diff)
    of.watch(
        output,
        lambda blob: print(
            "scatter_nd_update forward output: \n", blob.ndarray()
        ),
    )
    of.watch_diff(
        output, lambda blob: print("scatter_nd_update output_diff: \n", blob)
    )
    return output


def tf_train_scatter_nd_update(indices):
    x = tf.Variable(np.full((10,), 0.0, dtype=np.float32))
    const_x = tf.constant(0.0, shape=(10,), dtype=tf.float32)
    y = tf.Variable(np.full((5,), 2.0, dtype=np.float32))
    const_y = tf.constant(2.0, shape=(5,), dtype=tf.float32)

    with tf.GradientTape() as t1:
        # z1 = tf.compat.v1.scatter_nd_update(x, tf.Variable(indices), const_y)
        z1 = tf.tensor_scatter_nd_update(x, tf.Variable(indices), const_y)
    dz_dx = t1.gradient(z1, x)
    print("z1: \n", z1)
    print("dz_dx: \n", dz_dx)

    with tf.GradientTape() as t2:
        # Cannot use tf.compat.v1.scatter_nd_update because it's ref input
        # and don't work correctly with gradient and it cannot calculate
        # updates grad
        # z2 = tf.compat.v1.scatter_nd_update(x, tf.Variable(indices), y)
        z2 = tf.tensor_scatter_nd_update(x, tf.Variable(indices), y)
    dz_dy = t2.gradient(z2, y)
    print("z2: \n", z2)
    print("dz_dy: \n", dz_dy)

    return dz_dx, dz_dy


def test_case_1():
    r"""2D-indices scatter"""
    input = np.random.randint(1024, size=(10,)).astype(np.float32)
    # indices = np.random.randint(10, size=(5, 1)).astype(np.int32)
    indices = np.arange(10)
    np.random.shuffle(indices)
    indices = indices[:5].reshape(5, 1).astype(np.int32)
    updates = np.random.randint(1024, size=(5)).astype(np.float32)
    print("========== input: =========")
    print("input: \n", input)
    print("indices: \n", indices)
    print("updates: \n", updates)

    tf_out = tf.compat.v1.scatter_nd_update(
        tf.Variable(input), tf.Variable(indices), tf.Variable(updates)
    ).numpy()

    print("===========================")
    print("tf output: \n", tf_out)

    of_out = of_scatter_nd_update(input, indices, updates).ndarray()
    print("===========================")
    print("of output: \n", of_out)

    assert np.allclose(tf_out, of_out), "of_out is not equal to tf_out"
    print("==== test case 1 Done! ====")


def test_train():
    r"""scatter_nd_update grad test"""
    indices = np.arange(10)
    np.random.shuffle(indices)
    indices = indices[:5].reshape(5, 1).astype(np.int32)
    check_point = of.train.CheckPoint()
    check_point.init()
    scatter_nd_update_train_job(indices).get()

    global g_of_scatter_nd_update_input_diff
    global g_of_scatter_nd_update_updates_diff
    tf_dz_dx, tf_dz_dy = tf_train_scatter_nd_update(indices)
    of_input_grad = g_of_scatter_nd_update_input_diff
    of_updates_grad = g_of_scatter_nd_update_updates_diff
    tf_input_grad = tf_dz_dx.numpy()
    tf_updates_grad = tf_dz_dy.numpy()

    print("===========================")
    print("of scatter_nd_update input gradient: \n", of_input_grad)
    print("tf scatter_nd_update input gradient: \n", tf_input_grad)
    assert np.allclose(of_input_grad, tf_input_grad)
    print("tf scatter_nd_update updates gradient: \n", of_updates_grad)
    print("tf scatter_nd_update updates gradient: \n", tf_updates_grad)
    assert np.allclose(of_updates_grad, tf_updates_grad)
    print("==== test train Done ! ====")


if __name__ == "__main__":
    test_case_1()
    test_train()
