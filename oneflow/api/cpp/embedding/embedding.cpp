/*
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
*/
#include "oneflow/api/cpp/embedding/embedding.h"
#include "oneflow/core/embedding/embedding_manager.h"

namespace oneflow_api {
namespace embedding {

std::string CreateKeyValueStore(const std::string& key_value_store_options, int64_t local_rank_id,
                                int64_t rank_id, int64_t world_size) {
  oneflow::embedding::KeyValueStoreOptions options(key_value_store_options);
#ifdef WITH_CUDA
  oneflow::Singleton<oneflow::embedding::EmbeddingManager>::Get()->CreateKeyValueStore(
      options, local_rank_id, rank_id, world_size);
  return options.Name();
#else
  UNIMPLEMENTED() << "OneEmbedding Only Support with CUDA";
#endif
  return "";
}

void LoadSnapshot(const std::string& snapshot_name, const std::string& embedding_name,
                  int64_t local_rank_id, int64_t rank_id) {
#ifdef WITH_CUDA
  oneflow::Singleton<oneflow::embedding::EmbeddingManager>::Get()->LoadSnapshot(
      embedding_name, local_rank_id, rank_id, snapshot_name);
#else
  UNIMPLEMENTED() << "OneEmbedding Only Support with CUDA";
#endif
}

}  // namespace embedding
}  // namespace oneflow_api
