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

#include "oneflow/core/embedding/embedding.cuh"
#include "oneflow/core/embedding/hash_function.cuh"
#include "oneflow/core/job/embedding.h"

namespace oneflow {

namespace embedding {
int64_t GetEmbeddingVecSize(int64_t embedding_size, const std::string& optimizer) {
  int64_t embedding_vec_size = embedding_size;
  if (optimizer == "sgd") {
    // do nothing
  } else if (optimizer == "adam") {
    embedding_vec_size *= 3;
  } else {
    UNIMPLEMENTED();
  }
  return embedding_vec_size;
}

}  // namespace embedding

std::shared_ptr<embedding::Embedding<int64_t, float, XXH64, int32_t>>
EmbeddingMgr::GetEmbedding4Name(const std::string& name) {
  const auto& it = name2embeddings_.find(name);
  CHECK(it != name2embeddings_.end());
  return it->second;
}
void EmbeddingMgr::AddEmbeddingTable(const std::string& name, const std::string& path) {
  LOG(ERROR) << "AddEmbeddingTable start " << name;
  CHECK(name2embeddings_
            .emplace(name, std::make_shared<embedding::Embedding<int64_t, float, XXH64, int32_t>>(
                               30, 128, "tables", 12, 65536))
            .second);
  LOG(ERROR) << "AddEmbeddingTable end " << name;
}
}  // namespace oneflow
