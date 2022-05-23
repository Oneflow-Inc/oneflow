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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/global_for.h"

namespace py = pybind11;

namespace oneflow {

Maybe<void> EnableDTRStrategy(bool enable_dtr, size_t thres, int debug_level,
                              const std::string& heuristic) {
  CHECK_NOTNULL_OR_RETURN((Global<DTRConfig>::Get()));
  *Global<DTRConfig>::Get() = DTRConfig(enable_dtr, thres, debug_level, heuristic);
  CHECK_EQ_OR_RETURN(Global<vm::DtrCudaAllocator>::Get()->allocated_memory(), 0);
  Global<vm::DtrCudaAllocator>::Delete();
  // re-init the allocator using the new config
  Global<vm::DtrCudaAllocator>::SetAllocated(new vm::DtrCudaAllocator(0));
  return Maybe<void>::Ok();
}

ONEFLOW_API_PYBIND11_MODULE("dtr", m) {
  m.def("enable", &EnableDTRStrategy);
  m.def("is_enabled", &dtr::is_enabled);
  m.def("is_dtr_tensor", [](const std::shared_ptr<one::Tensor>& tensor) -> bool {
    return std::dynamic_pointer_cast<one::DTRMirroredTensor>(tensor) != nullptr;
  });
  m.def("is_in_memory", [](const std::shared_ptr<one::Tensor>& tensor) -> Maybe<bool> {
    auto dtr_tensor = std::dynamic_pointer_cast<one::DTRMirroredTensor>(tensor);
    CHECK_NOTNULL_OR_RETURN(dtr_tensor);
    return dtr_tensor->is_in_memory();
  });
  m.def("allocated_memory",
        []() -> size_t { return Global<vm::DtrCudaAllocator>::Get()->allocated_memory(); });
  m.def("display_all_pieces",
        []() -> void { return Global<vm::DtrCudaAllocator>::Get()->DisplayAllPieces(); });
  m.def("pool_display", []() -> Maybe<void> { return Global<dtr::TensorPool>::Get()->display(); });
  m.def("pool_verbose_display",
        []() -> Maybe<void> { return Global<dtr::TensorPool>::Get()->verbose_display(); });
  m.def("set_non_evictable", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    auto dtr_tensor =
        std::dynamic_pointer_cast<one::DTRMirroredTensor>(JUST(t->AsMirroredTensor()));
    CHECK_NOTNULL_OR_RETURN(dtr_tensor);
    std::dynamic_pointer_cast<vm::DTREagerBlobObject>(JUST(dtr_tensor->eager_blob_object()))
        ->set_evictable(false);
    return Maybe<void>::Ok();
  });
  m.def("tensor_info", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<std::string> {
    auto dtr_tensor =
        std::dynamic_pointer_cast<one::DTRMirroredTensor>(JUST(t->AsMirroredTensor()));
    CHECK_NOTNULL_OR_RETURN(dtr_tensor);
    auto dtr_ebo =
        std::dynamic_pointer_cast<vm::DTREagerBlobObject>(JUST(dtr_tensor->eager_blob_object()));
    std::stringstream ss;
    ss << "tensor: " << dtr_tensor.get() << ", compute op: " << dtr_ebo->compute_op_type_name()
       << ", is_in_memory: " << dtr_ebo->is_in_memory()
       << ", is_evictable: " << dtr_ebo->is_evictable() << ", pinned: " << dtr_ebo->num_pinned()
       << ", id: " << dtr_ebo->id();
    return ss.str();
  });
  m.def("evict", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    auto dtr_tensor =
        std::dynamic_pointer_cast<one::DTRMirroredTensor>(JUST(t->AsMirroredTensor()));
    CHECK_NOTNULL_OR_RETURN(dtr_tensor);
    JUST(std::dynamic_pointer_cast<vm::DTREagerBlobObject>(JUST(dtr_tensor->eager_blob_object()))
             ->evict(false));
    return Maybe<void>::Ok();
  });
  m.def("is_evictable", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<bool> {
    if (auto dtr_tensor =
            std::dynamic_pointer_cast<one::DTRMirroredTensor>(JUST(t->AsMirroredTensor()))) {
      return std::dynamic_pointer_cast<vm::DTREagerBlobObject>(
                 JUST(dtr_tensor->eager_blob_object()))
          ->is_evictable();
    }
    return false;
  });
}

}  // namespace oneflow
