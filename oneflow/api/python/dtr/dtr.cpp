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

Maybe<bool> IsDTREnabled() {
  CHECK_NOTNULL_OR_RETURN((Global<DTRConfig>::Get()));
  return Global<DTRConfig>::Get()->is_enabled;
}

void ApiEnableDTRStrategy(bool enable_dtr, size_t thres, int debug_level,
                          const std::string& heuristic) {
  EnableDTRStrategy(enable_dtr, thres, debug_level, heuristic).GetOrThrow();
}

bool ApiIsDTREnabled() { return IsDTREnabled().GetOrThrow(); }

ONEFLOW_API_PYBIND11_MODULE("dtr", m) {
  m.def("enable", &ApiEnableDTRStrategy);
  m.def("is_enabled", &ApiIsDTREnabled);
  m.def("allocated_memory",
        []() -> size_t { return Global<vm::DtrCudaAllocator>::Get()->allocated_memory(); });
  m.def("display_all_pieces",
        []() -> void { return Global<vm::DtrCudaAllocator>::Get()->DisplayAllPieces(); });
  m.def("display", []() -> void { Global<one::DTRTensorPool>::Get()->display().GetOrThrow(); });
  m.def("set_non_evictable", [](const std::shared_ptr<one::Tensor> t) -> void {
    if (auto dtr_tensor =
            std::dynamic_pointer_cast<one::DTRMirroredTensor>(CHECK_JUST(t->AsMirroredTensor()))) {
      std::dynamic_pointer_cast<vm::DTREagerBlobObject>(CHECK_JUST(dtr_tensor->eager_blob_object()))
          ->set_evict_attr(false);
    }
  });
  m.def("evict", [](const std::shared_ptr<one::Tensor> t) -> void {
    if (auto dtr_tensor =
            std::dynamic_pointer_cast<one::DTRMirroredTensor>(CHECK_JUST(t->AsMirroredTensor()))) {
      CHECK_JUST(std::dynamic_pointer_cast<vm::DTREagerBlobObject>(
                     CHECK_JUST(dtr_tensor->eager_blob_object()))
                     ->evict());
    } else {
      CHECK(false);
    }
  });
  m.def("is_evictable", [](const std::shared_ptr<one::Tensor> t) -> bool {
    if (auto dtr_tensor =
            std::dynamic_pointer_cast<one::DTRMirroredTensor>(CHECK_JUST(t->AsMirroredTensor()))) {
      return std::dynamic_pointer_cast<vm::DTREagerBlobObject>(
                 CHECK_JUST(dtr_tensor->eager_blob_object()))
          ->is_evictable();
    }
    return false;
  });
}

}  // namespace oneflow
