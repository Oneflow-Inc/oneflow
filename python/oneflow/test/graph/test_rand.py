import oneflow as flow
import oneflow.nn as nn

placement = flow.placement("cuda", {0: [0, 1]})


class TrainGraph(flow.nn.Graph):
    def __init__(self,):
        super().__init__()

    def build(self):
        ids = flow.randint(
            0, 150000, (65536, 26), placement=placement, sbp=flow.sbp.split(0), dtype=flow.int64)
        return ids

graph = TrainGraph()
for i in range(1):
    ids = graph()
    print(ids.shape, ids.sbp, ids.placement)
