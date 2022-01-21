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
  OneEmbeddingHandler(const std::string embedding_option_string){
    embedding_option_.reset(new embedding::EmbeddingOptions(embedding_option_string));
    embedding_name_ = embedding_option_->EmbeddingName(); 
    // WARNING: hard code here
    CreateKeyValueStore(*embedding_option_, 0, 1, 0);
  }  	
  
  void LoadSnapshot(int64_t parallel_id, const std::string snapshot_name){
    Global<EmbeddingMgr>::Get()->LoadSnapshot(embedding_name_, parallel_id, snapshot_name);
  }
  
  void SaveSnapshot(int64_t parallel_id, const std::string snapshot_name){
    Global<EmbeddingMgr>::Get()->SaveSnapshot(embedding_name_, parallel_id, snapshot_name); 
  }
  
  void CreateKeyValueStore(const embedding::EmbeddingOptions& embedding_option, int64_t parallel_id, int64_t parallel_num, uint64_t cuda_device_id){
    Global<EmbeddingMgr>::Get()->CreateKeyValueStore(embedding_option, parallel_id, parallel_num, cuda_device_id); 
  }  
  
  private: 
    std::string embedding_name_;
    std::unique_ptr<embedding::EmbeddingOptions> embedding_option_; 
}; 


ONEFLOW_API_PYBIND11_MODULE("", m){
  py::class_<OneEmbeddingHandler, std::shared_ptr<OneEmbeddingHandler>>(m, "OneEmbeddingHandler")
      .def(py::init([](const std::string& embedding_option_str){return std::make_shared<OneEmbeddingHandler>(embedding_option_str); })) 
      .def("SaveSnapshot", &OneEmbeddingHandler::SaveSnapshot)
      .def("LoadSnapshot", &OneEmbeddingHandler::LoadSnapshot)
      .def("CreateKeyValueStore", &OneEmbeddingHandler::CreateKeyValueStore); 
}

}  // namespace oneflow
