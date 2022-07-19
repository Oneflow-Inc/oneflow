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
#include "oneflow/api/cpp/one_embedding/one_embedding.h"
#include "oneflow/core/embedding/embedding_manager.h"

namespace oneflow_api {
namespace one_embedding {

OneEmbeddingHandler::OneEmbeddingHandler(const std::string& key_value_store_option_string) {
  oneflow::embedding::KeyValueStoreOptions key_value_store_options(key_value_store_option_string);
  embedding_name_ = key_value_store_options.Name();
  CreateKeyValueStore(key_value_store_options);
}

void OneEmbeddingHandler::CreateKeyValueStore(
    const oneflow::embedding::KeyValueStoreOptions& key_value_store_options) {
#ifdef WITH_CUDA
  oneflow::Singleton<oneflow::embedding::EmbeddingManager>::Get()->CreateKeyValueStore(
      key_value_store_options, 0, 0, 1);
#else
  UNIMPLEMENTED() << "Only Support with CUDA";
#endif
}

void OneEmbeddingHandler::LoadSnapshot(const std::string& snapshot_name) {
#ifdef WITH_CUDA
  oneflow::Singleton<oneflow::embedding::EmbeddingManager>::Get()->LoadSnapshot(embedding_name_, 0, 0,
                                                                       snapshot_name);
#else
  UNIMPLEMENTED() << "Only Support with CUDA";
#endif
}

}  // namespace one_embedding
}  // namespace oneflow_api
