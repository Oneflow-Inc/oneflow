import oneflow as flow

P = flow.placement("cuda", ranks=[0, 1])
x = flow.randn(16, 16, 16**2*3)
x.to_global(placement=P, sbp=flow.sbp.broadcast)

def random_masking(x, mask_ratio=0.75):
    N, L, _ = x.shape
    len_keep = int(L * (1 - mask_ratio))

    mask = flow.ones([N, L], dtype=x.dtype, placement=P, sbp=flow.sbp.broadcast)
    print(f"len_keep {len_keep}")
	#mask[:, :4] = 0
    mask[:, :len_keep] = 0
    return mask

class Graph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m=random_masking

    def build(self, x):
        return self.m(x)

graph = Graph()
graph.debug(2)
print(graph(x))

# EPOCH_NUM=100
# for i in range(EPOCH_NUM):
#     print(graph(x))