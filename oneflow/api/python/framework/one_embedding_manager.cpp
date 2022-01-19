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
  void SaveSnapshot(const std::string& embedding_name, int64_t parallel_id, const std::string& snapshot_name){
    Global<EmbeddingMgr>::Get()->SaveSnapshot(embedding_name, parallel_id, snapshot_name); 
  }
  
}; 

/*
ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<EmbeddingMgr, std::shared_ptr<EmbeddingMgr>>(m, "OneEmbeddingManager")
      .def(py::init([]() { return std::make_shared<EmbeddingMgr>(); })) // todo: remove
      .def("GetKeyValueStore", &EmbeddingMgr::GetKeyValueStore)
      .def("GetOrCreateKeyValueStore", &EmbeddingMgr::GetOrCreateKeyValueStore)
      .def("CreateKeyValueStore", &EmbeddingMgr::CreateKeyValueStore)
      .def("SaveSnapshot", &EmbeddingMgr::SaveSnapshot)
      .def("LoadSnapshot", &EmbeddingMgr::LoadSnapshot);
}
*/

ONEFLOW_API_PYBIND11_MODULE("", m){
  py::class_<OneEmbeddingHandler, std::shared_ptr<OneEmbeddingHandler>>(m, "OneEmbeddingManager")
      .def("SaveSnapshot", &OneEmbeddingHandler::SaveSnapshot); 
}

}  // namespace oneflow
