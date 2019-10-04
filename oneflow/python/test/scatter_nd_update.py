import oneflow as of
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.enable_eager_execution()
# assert(tf.executing_eagerly())


def of_scatter_nd_update_job(input, indices, updates):
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


def test_case_1():
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

    of_out = of_scatter_nd_update_job(input, indices, updates).ndarray()
    print("===========================")
    print("of output: \n", of_out)

    assert np.allclose(tf_out, of_out), "of_out is not equal to tf_out"
    print("==== test case 1 Done! ====")


if __name__ == "__main__":
    test_case_1()
