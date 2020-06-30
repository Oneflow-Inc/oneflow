import numpy as np
import oneflow as flow


def test_watch_scope(test_case):
    def Print(prefix):
        def _print(x):
            print(prefix)
            print(x)

        return _print

    blob_watched = {}

    @flow.global_function(flow.FunctionConfig())
    def ReluJob(x=flow.FixedTensorDef((2, 5))):
        with flow.watch_scope(blob_watched):
            y = flow.nn.relu(x)
            z = flow.nn.relu(y)
            return z

    index = [-2]
    data = []
    x = np.ones((2, 5), dtype=np.float32)
    ReluJob(x).get()
    for lbn, blob_data in blob_watched.items():
        blob_data["blob_def"].location
        blob_data["blob"]
