import oneflow.python.framework.ofblob as ofblob
import oneflow.python.framework.oneflow_internal as oneflow_internal

class PythonCallback(oneflow_internal.ForeignCallback):
    def __init__(self):
        TODO()

    def push_blob(self, of_blob):
        assert False, "UNIMPLEMENTED"

    def pull_blob(self, of_blob):
        assert False, "UNIMPLEMENTED"

    def finish(self):
        assert False, "UNIMPLEMENTED"

    def PushBlob(self, of_blob_ptr):
        self.push_blob(ofblob.OfBlob(of_blob_ptr))

    def PullBlob(self, of_blob_ptr):
        self.pull_blob(ofblob.OfBlob(of_blob_ptr))

    def Finish(self):
        self.finish()
