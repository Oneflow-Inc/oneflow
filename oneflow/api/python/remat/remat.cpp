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
#include "oneflow/core/vm/remat/allocator.h"
#include "oneflow/core/vm/remat/env.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/eager/tensor_storage.h"

namespace py = pybind11;

namespace oneflow {

namespace {
Maybe<vm::RematableTensorStorage> rematable_storage(const std::shared_ptr<one::Tensor>& tensor) {
  auto ret = std::dynamic_pointer_cast<vm::RematableTensorStorage>(
      JUST(tensor->eager_blob_object())->tensor_storage());
  CHECK_NOTNULL_OR_RETURN(ret);
  return ret;
}
}  // namespace

ONEFLOW_API_PYBIND11_MODULE("remat", m) {
  m.def("is_in_memory", [](const std::shared_ptr<one::Tensor>& tensor) -> Maybe<bool> {
    return JUST(rematable_storage(tensor))->is_in_memory();
  });
  m.def("allocated_memory", [](const std::string& device_str) -> Maybe<size_t> {
    auto device = JUST(Device::ParseAndNew(device_str));
    return Singleton<remat::AllocatorManager>::Get()
        ->CreateOrGetAllocator(device->enum_type(), device->device_id())
        ->allocated_memory();
  });
  m.def("display", [](const std::string& device_str) -> Maybe<void> {
    auto device = JUST(Device::ParseAndNew(device_str));
    Singleton<remat::AllocatorManager>::Get()
        ->CreateOrGetAllocator(device->enum_type(), device->device_id())
        ->DisplayAllPieces();
    return Maybe<void>::Ok();
  });
  m.def("remat", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    // TODO: an instruction
    JUST(rematable_storage(t))->Remat();
    return Maybe<void>::Ok();
  });
  m.def("evict", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    // TODO: an instruction
    JUST(rematable_storage(t))->Evict(false);
    return Maybe<void>::Ok();
  });
  m.def("is_evictable", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<bool> {
    return JUST(rematable_storage(t))->is_evictable();
  });
  m.def("disable_eviction", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    JUST(rematable_storage(t))->set_eviction_disabled(true);
    return Maybe<void>::Ok();
  });
  m.def("clear_compute_op", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    JUST(rematable_storage(t))->clear_compute_op();
    return Maybe<void>::Ok();
  });
  m.def("clear_stats", []() { Singleton<remat::Env>::Get()->clear_stats(); });
  m.def("forced_eviction_num",
        []() { return Singleton<remat::Env>::Get()->forced_eviction_num(); });
  m.def("eager_eviction_num", []() { return Singleton<remat::Env>::Get()->eager_eviction_num(); });
  m.def("recomputation_num", []() { return Singleton<remat::Env>::Get()->recomputation_num(); });
  m.def("set_budget_in_bytes", [](size_t budget_in_bytes) {
    Singleton<remat::Env>::Get()->set_budget_in_bytes(budget_in_bytes);
  });
  m.def("budget_in_bytes", []() { return Singleton<remat::Env>::Get()->budget_in_bytes(); });
  m.def("set_small_pieces_optimization", [](bool enabled) {
    return Singleton<remat::Env>::Get()->set_small_pieces_optimization(enabled);
  });
  m.def("is_small_pieces_optimization_enabled",
        []() { return Singleton<remat::Env>::Get()->is_small_pieces_optimization_enabled(); });
}

}  // namespace oneflow
