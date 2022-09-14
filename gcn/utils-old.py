import numpy as np
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./dataset/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj.tocoo()

    idx_train = np.array(range(140), dtype=np.int32)
    idx_val = np.array(range(200, 500), dtype=np.int32)
    idx_test = np.array(range(500, 1500), dtype=np.int32)

    x = np.array(features.todense(), dtype=np.float32)
    labels = np.where(labels)[1].astype(np.int32)

    cooRowInd = adj.row
    cooColInd = adj.col
    cooValues = adj.data.astype(np.float32)

    num_classes = 7
    print(x.shape)
    return (
        (cooRowInd, cooColInd, cooValues),
        x,
        labels,
        idx_train,
        idx_val,
        idx_test,
        num_classes,
    )


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test, num_classes = load_data()
