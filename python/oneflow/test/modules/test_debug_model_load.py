"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np


parallel_num = 1
emb_size = 16
ids = []
partition_index = []
cur_rank_unique_ids = []
num_unique_ids = []
cur_rank_num_unique_ids = []
cur_rank_reverse_idx = []
ids_reverse_idx = []
num_unique_ids_matrix = []
embeddings = []
cur_rank_embeddings = []
embedding_diff = []
cur_rank_unique_embedding_diff = []

for parallel_id in range(parallel_num):
    ids.append(np.fromfile("test/ids_" + str(parallel_id), dtype=np.int64))
    partition_index.append(
        np.fromfile("test/partition_index_" + str(parallel_id), dtype=np.int32)
    )
    cur_rank_unique_ids.append(
        np.fromfile("test/cur_rank_unique_ids_" + str(parallel_id), dtype=np.int64)
    )
    num_unique_ids.append(
        np.fromfile("test/num_unique_ids_" + str(parallel_id), dtype=np.int32)
    )
    cur_rank_num_unique_ids.append(
        np.fromfile("test/cur_rank_num_unique_ids_" + str(parallel_id), dtype=np.int32)
    )
    cur_rank_reverse_idx.append(
        np.fromfile("test/cur_rank_reverse_idx_" + str(parallel_id), dtype=np.int32)
    )
    ids_reverse_idx.append(
        np.fromfile("test/ids_reverse_idx_" + str(parallel_id), dtype=np.int32)
    )
    num_unique_ids_matrix.append(
        np.fromfile("test/num_unique_ids_matrix_" + str(parallel_id), dtype=np.int32)
    )
    embeddings.append(
        np.fromfile("test/embeddings_" + str(parallel_id), dtype=np.float32)
    )
    cur_rank_embeddings.append(
        np.fromfile("test/cur_rank_embeddings_" + str(parallel_id), dtype=np.float32)
    )
    embedding_diff.append(
        np.fromfile("test/embedding_diff_" + str(parallel_id), dtype=np.float32)
    )
    cur_rank_unique_embedding_diff.append(
        np.fromfile(
            "test/cur_rank_unique_embedding_diff_" + str(parallel_id), dtype=np.float32
        )
    )


np_global_ids = np.concatenate((ids))


# make np_unique_ids_list
np_unique_ids_list = []
np_model = np.load(
    "/data/guoran/models/RecommenderSystems/dlrm/initial_parameters/embedding_weight.npy"
)
for i in range(parallel_num):
    np_ids = ids[i]
    np_unique_ids, np_ids_reverse_idx = np.unique(np_ids, return_inverse=True)
    np_unique_ids_list.append(np_unique_ids)
    print(
        i, "ids_reverse_idx: ", np.array_equal(ids_reverse_idx[i], np_ids_reverse_idx)
    )
    print(i, "num_unique_ids: ", num_unique_ids[i] == np_unique_ids.size)

    np_embedding = np.zeros((np_ids.size, emb_size))
    for j in range(np_ids.size):
        np_embedding[j, :] = np_model[np_ids[j]]
    print("np_embedding", np_embedding)
    print("embeddings", embeddings[i].reshape(-1, 16))
    print(
        i,
        "embedding: ",
        np.array_equal(embeddings[i], np_embedding.flatten(),),
        ", to compare embedding, please set init value to key[i] instead of random value",
    )


# unittest
for i in range(parallel_num):
    cur_rank_partition = []
    cur_rank_num_reverse_ids = 0

    for j in range(parallel_num):
        np_unique_ids = np_unique_ids_list[j]
        cur_rank_partition += np_unique_ids[
            np.where(np_unique_ids % parallel_num == i)
        ].tolist()
        cur_rank_num_reverse_ids += num_unique_ids_matrix[0][j * parallel_num + i]

    np_cur_rank_unique_ids, np_cur_rank_reverse_idx = np.unique(
        np.array(cur_rank_partition), return_inverse=True
    )
    print(
        i,
        "cur_rank_ids_reverse_idx: ",
        np.array_equal(
            cur_rank_reverse_idx[i][0:cur_rank_num_reverse_ids], np_cur_rank_reverse_idx
        ),
        ", cur_rank_ids_reverse_idx not need to be True, because partition is not sorted",
    )

    print(
        i,
        "cur_rank_unique_ids: ",
        np.array_equal(
            cur_rank_unique_ids[i][0 : cur_rank_num_unique_ids[i][0]],
            np_cur_rank_unique_ids,
        ),
    )
    print(
        i,
        "cur_rank_num_unique_ids: ",
        cur_rank_num_unique_ids[i] == np_cur_rank_unique_ids.size,
    )

    np_embedding_diff = embedding_diff
    np_cur_rank_unique_embedding_diff = np.zeros(
        cur_rank_unique_embedding_diff[i].size
    ).reshape(-1, emb_size)
    # for k in range(cur_rank_num_unique_ids[i][0]):
    #    np_cur_rank_unique_embedding_diff[k, :] = sum(
    #        np_embedding_diff[0][np.where(ids[0] == cur_rank_unique_ids[i][k])[0] * emb_size]
    #    ) + sum(
    #        np_embedding_diff[1][np.where(ids[1] == cur_rank_unique_ids[i][k])[0] * emb_size]
    #    )
#
# print(
#    "cur_rank",
#    cur_rank_unique_embedding_diff[i],
#    np.where(cur_rank_unique_embedding_diff[i] > 1)[0].size,
# )
# print(
#    i,
#    "cur_rank_unique_embedding_diff",
#    np.array_equal(
#        np_cur_rank_unique_embedding_diff.flatten(),
#        cur_rank_unique_embedding_diff[i],
#    ),
# )
