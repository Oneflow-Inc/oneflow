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
#ifndef ONEFLOW_CORE_JOB_EMBEDDING_H_
#define ONEFLOW_CORE_JOB_EMBEDDING_H_

#include "oneflow/core/device/cuda_util.h"

struct XXH64;

namespace oneflow {

namespace embedding {

int64_t GetEmbeddingVecSize(int64_t embedding_size, const std::string& optimizer);

template<typename Key, typename Elem, typename Hash, typename Idx>
class Embedding;

}  // namespace embedding

class EmbeddingMgr final {
 public:
  EmbeddingMgr() = default;
  ~EmbeddingMgr() = default;

  std::shared_ptr<embedding::Embedding<int64_t, float, XXH64, int32_t>> GetEmbedding4Name(
      const std::string& name);
  void AddEmbeddingTable(const std::string& name, const std::string& path);

 private:
  HashMap<std::string, std::shared_ptr<embedding::Embedding<int64_t, float, XXH64, int32_t>>>
      name2embeddings_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_CLUSTER_DESC_H_
