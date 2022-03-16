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
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/embedding/embedding_manager.h"
namespace py = pybind11;

namespace oneflow {

class OneEmbeddingHandler final {
 public:
  OneEmbeddingHandler(const std::string& key_value_store_option_string, int64_t local_rank_id,
                      int64_t rank_id, int64_t world_size)
      : local_rank_id_(local_rank_id), rank_id_(rank_id), world_size_(world_size) {
    embedding::KeyValueStoreOptions key_value_store_options(key_value_store_option_string);
    embedding_name_ = key_value_store_options.Name();
    CreateKeyValueStore(key_value_store_options);
  }

  void LoadSnapshot(const std::string& snapshot_name) {
#ifdef WITH_CUDA
    Global<embedding::EmbeddingManager>::Get()->LoadSnapshot(embedding_name_, local_rank_id_,
                                                             rank_id_, snapshot_name);
#else
    UNIMPLEMENTED() << "Only Support with CUDA";
#endif
  }

  void SaveSnapshot(const std::string& snapshot_name) {
#ifdef WITH_CUDA
    Global<embedding::EmbeddingManager>::Get()->SaveSnapshot(embedding_name_, local_rank_id_,
                                                             rank_id_, snapshot_name);
#else
    UNIMPLEMENTED() << "Only Support with CUDA";
#endif
  }

 private:
  void CreateKeyValueStore(const embedding::KeyValueStoreOptions& key_value_store_options) {
#ifdef WITH_CUDA
    Global<embedding::EmbeddingManager>::Get()->CreateKeyValueStore(
        key_value_store_options, local_rank_id_, rank_id_, world_size_);
#else
    UNIMPLEMENTED() << "Only Support with CUDA";
#endif
  }

  std::string embedding_name_;
  int64_t local_rank_id_;
  int64_t rank_id_;
  int64_t world_size_;
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OneEmbeddingHandler, std::shared_ptr<OneEmbeddingHandler>>(m, "OneEmbeddingHandler")
      .def(py::init([](const std::string& key_value_store_option_str, const int64_t local_rank_id,
                       const int64_t rank_id, const int64_t world_size) {
        return std::make_shared<OneEmbeddingHandler>(key_value_store_option_str, local_rank_id,
                                                     rank_id, world_size);
      }))
      .def("SaveSnapshot", &OneEmbeddingHandler::SaveSnapshot)
      .def("LoadSnapshot", &OneEmbeddingHandler::LoadSnapshot);
}

}  // namespace oneflow
