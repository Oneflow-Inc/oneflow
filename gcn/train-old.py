import oneflow as flow
import oneflow.typing as tp

from utils import load_data

adj, features, labels, idx_train, idx_val, idx_test, num_classes = load_data()
cooRowInd, cooColInd, cooValues = adj

NODE_NUM, FEATURE_DIM = features.shape
rows = NODE_NUM
cols = NODE_NUM
BATCH_SIZE = NODE_NUM
HIDDEN_SIZE = 16

EDGE_NUM = len(cooRowInd)
TRAIN_NUM = len(idx_train)


@flow.global_function(type="train")
def train_job(
    x: tp.Numpy.Placeholder((BATCH_SIZE, FEATURE_DIM), dtype=flow.float32),  # noqa
    cooRowInd: tp.Numpy.Placeholder((EDGE_NUM,), dtype=flow.int32),  # noqa
    cooColInd: tp.Numpy.Placeholder((EDGE_NUM,), dtype=flow.int32),  # noqa
    cooValues: tp.Numpy.Placeholder((EDGE_NUM,), dtype=flow.float32),  # noqa
    train_indices: tp.Numpy.Placeholder((TRAIN_NUM,), dtype=flow.int32),  # noqa
    train_labels: tp.Numpy.Placeholder((TRAIN_NUM,), dtype=flow.int32),  # noqa
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        """conv1 = GCNConv(dataset.num_node_features, 16)
        x = self.conv1(x, edge_index)
        x = F.relu(x)"""
        x = flow.layers.GraphConvolution(
            x, cooRowInd, cooColInd, cooValues, rows, cols, FEATURE_DIM, HIDDEN_SIZE
        )
        # x = flow.layers.dense(x, HIDDEN_SIZE, activation=flow.nn.relu, name="fc1",
        #                       kernel_initializer=flow.random_uniform_initializer())

        """x = F.dropout(x, training=self.training)"""
        x = flow.nn.dropout(x, rate=0.5)

        """self.conv2 = GCNConv(16, num_classes)
        x = self.conv2(x, edge_index)"""
        # x = flow.layers.GraphConvolution(x, cooRowInd, cooColInd, cooValues, rows, cols, HIDDEN_SIZE, num_classes)
        x = flow.layers.dense(
            x,
            num_classes,
            activation=flow.nn.relu,
            name="fc2",
            kernel_initializer=flow.random_uniform_initializer(),
        )

        """return F.log_softmax(x, dim=1)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        """
        x = flow.gather(params=x, indices=train_indices, axis=0)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(train_labels, x)  # noqa

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss  # noqa


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    train_indices = idx_train
    train_labels = labels[idx_train]

    for epoch in range(50):
        loss = train_job(
            features, cooRowInd, cooColInd, cooValues, train_indices, train_labels
        )
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, 50, loss.mean()))
