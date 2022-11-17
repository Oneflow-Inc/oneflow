import numpy as np
import oneflow as flow
from gcn_spmm import GCN


tables = [
    flow.one_embedding.make_table_options(
        flow.one_embedding.make_uniform_initializer(low=-0.1, high=0.1)
    )
]

store_options = flow.one_embedding.make_cached_ssd_store_options(
    cache_budget_mb=8142,
    persistent_path="mlp_embedding", 
    capacity=40000000,
    size_factor=1,              
    physical_block_size=512
)

embedding_size = 16
embedding = flow.one_embedding.MultiTableEmbedding(
    name="mlp_embedding",
    embedding_dim=embedding_size,
    dtype=flow.float,
    key_type=flow.int64,
    tables=tables,
    store_options=store_options,
)

embedding.to("cuda")

""" [n, 8] = [n, 16] (embedding, uniform_initializer) x [8, 16]^T
[n, 4] = [n, 8] x [4, 8]^T  """

mlp = flow.nn.FusedMLP(
    in_features=embedding_size, # 16
    hidden_features=[8],
    out_features=4,
    skip_final_activation=True,
)
mlp.to("cuda")
for name, param in mlp.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())
print("-------------------")

class TrainGraph(flow.nn.Graph):
    def __init__(self,):
        super().__init__()
        self.embedding_lookup = embedding
        self.mlp = mlp
        self.add_optimizer(
            flow.optim.SGD(self.embedding_lookup.parameters(), lr=0.1, momentum=0.0)
        )
        self.add_optimizer(
            flow.optim.SGD(self.mlp.parameters(), lr=0.1, momentum=0.0)
        )
    def build(self, ids):
        embedding = self.embedding_lookup(ids)
        loss = self.mlp(flow.reshape(embedding, (-1,  embedding_size)))
        loss = loss.sum()
        loss.backward()
        return loss


ids = np.random.randint(0, 1000, (2, 1), dtype=np.int64)
ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
graph = TrainGraph()
loss = graph(ids_tensor)
print(ids_tensor, loss)


""" with flow.one_embedding.make_persistent_table_writer(["mlp_embedding/0-1"], "saved_snapshot", flow.int32, flow.float, embedding_size, 512) as writer:
    keys = np.arange(8).astype(np.int32)
    values = np.arange(8).astype(np.float)
    writer.write(keys, values)  """
