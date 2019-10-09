import oneflow
import numpy


class BlobDump(object):
    def __init__(self, blob):
        assert str(type(blob)) == "<class 'oneflow.python.framework.remote_blob.RemoteBlob'>"
        self.blob_ = blob

    def __call__(self):
        oneflow.watch(self.blob_, lambda blob: self.dump(blob.ndarray()))

    def dump(self, ndarray):
        numpy.save(self.blob_.op_name + "-" + self.blob_.blob_name, ndarray)
