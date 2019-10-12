import oneflow
import numpy


class BlobDump(object):
    def __init__(self, blob):
        assert (
            str(type(blob))
            == "<class 'oneflow.python.framework.remote_blob.RemoteBlob'>"
        )
        self.blob_ = blob

    def __call__(self, dump_diff=False):
        oneflow.watch(self.blob_, lambda blob: self.dump(blob.ndarray()))

    def dump(self, ndarray):
        numpy.save(self.blob_.op_name + "-" + self.blob_.blob_name, ndarray)


def print_runtime_blob_shape(blob, print_diff=False, name=None):
    if name is None:
        name = ""

    def print_cb(of_blob):
        fmt = "backward {} shape: " if print_diff else "{} shape: "
        print(
            fmt.format(name if name else blob.op_name + " - " + blob.blob_name),
            of_blob.ndarray().shape,
        )

    if print_diff:
        oneflow.watch_diff(blob, print_cb)
    else:
        oneflow.watch(blob, print_cb)
