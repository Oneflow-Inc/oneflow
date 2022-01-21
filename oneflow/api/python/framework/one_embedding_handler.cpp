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
  OneEmbeddingHandler(const std::string& embedding_option_string, int64_t rank_id, int64_t world_size):rank_id_(rank_id), world_size_(world_size){
    embedding_option_.reset(new embedding::EmbeddingOptions(embedding_option_string));
    embedding_name_ = embedding_option_->EmbeddingName(); 
    CreateKeyValueStore(*embedding_option_, rank_id_, world_size_);
  }  	
  
  void LoadSnapshot(const std::string& snapshot_name){
    Global<EmbeddingMgr>::Get()->LoadSnapshot(embedding_name_, rank_id_, snapshot_name);
  }
  
  void SaveSnapshot(const std::string& snapshot_name){
    Global<EmbeddingMgr>::Get()->SaveSnapshot(embedding_name_, rank_id_, snapshot_name); 
  }
  
  void CreateKeyValueStore(const embedding::EmbeddingOptions& embedding_option, int64_t num_rank, int64_t world_size){
    Global<EmbeddingMgr>::Get()->CreateKeyValueStore(embedding_option, rank_id_, world_size_); 
  }  
  
  private: 
    std::string embedding_name_;
    std::unique_ptr<embedding::EmbeddingOptions> embedding_option_; 
    int64_t rank_id_; 
    int64_t world_size_;  
}; 


ONEFLOW_API_PYBIND11_MODULE("", m){
  py::class_<OneEmbeddingHandler, std::shared_ptr<OneEmbeddingHandler>>(m, "OneEmbeddingHandler")
      .def(py::init([](const std::string& embedding_option_str, const int64_t rank_id, const int64_t world_size){return std::make_shared<OneEmbeddingHandler>(embedding_option_str, rank_id, world_size); })) 
      .def("SaveSnapshot", &OneEmbeddingHandler::SaveSnapshot)
      .def("LoadSnapshot", &OneEmbeddingHandler::LoadSnapshot)
      .def("CreateKeyValueStore", &OneEmbeddingHandler::CreateKeyValueStore); 
}

}  // namespace oneflow
