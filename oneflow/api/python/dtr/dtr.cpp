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
#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/eager/tensor_storage.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("dtr", m) {
  m.def("is_enabled", &dtr::is_enabled);
  m.def("is_in_memory", [](const std::shared_ptr<one::Tensor>& tensor) -> Maybe<bool> {
    return JUST(tensor->eager_blob_object())->tensor_storage()->is_in_memory();
  });
  m.def("allocated_memory", [](const std::string& device_str) -> Maybe<size_t> {
    auto device = JUST(Device::ParseAndNew(device_str));
    return Singleton<dtr::AllocatorManager>::Get()
        ->CreateOrGetAllocator(device->enum_type(), device->device_id())
        ->allocated_memory();
  });
  m.def("display", [](const std::string& device_str) -> Maybe<void> {
    auto device = JUST(Device::ParseAndNew(device_str));
    Singleton<dtr::AllocatorManager>::Get()
        ->CreateOrGetAllocator(device->enum_type(), device->device_id())
        ->DisplayAllPieces();
    return Maybe<void>::Ok();
  });
  // m.def("display_all_pieces",
  //       []() -> void { return Global<vm::DtrCudaAllocator>::Get()->DisplayAllPieces(); });
  // m.def("clear_num_ops_and_set_first", []() -> void {
  //   Global<dtr::TensorPool>::Get()->clear_num_ops();
  //   Global<vm::DtrCudaAllocator>::Get()->first_time = true;
  // });
  // m.def("get_num_ops", []() -> int { return Global<dtr::TensorPool>::Get()->num_ops(); });
  // m.def("pool_display", []() -> Maybe<void> { return
  // Global<dtr::TensorPool>::Get()->display(); }); m.def("pool_verbose_display",
  //       []() -> Maybe<void> { return Global<dtr::TensorPool>::Get()->verbose_display(); });
  // m.def("set_non_evictable", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
  //   auto dtr_tensor =
  //       std::dynamic_pointer_cast<one::DTRMirroredTensor>(JUST(t->AsMirroredTensor()));
  //   CHECK_NOTNULL_OR_RETURN(dtr_tensor);
  //   std::dynamic_pointer_cast<vm::DTREagerBlobObject>(JUST(dtr_tensor->eager_blob_object()))
  //       ->set_evictable(false);
  //   return Maybe<void>::Ok();
  // });
  // m.def("tensor_info", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<std::string> {
  //   auto dtr_tensor =
  //       std::dynamic_pointer_cast<one::DTRMirroredTensor>(JUST(t->AsMirroredTensor()));
  //   CHECK_NOTNULL_OR_RETURN(dtr_tensor);
  //   auto dtr_ebo =
  //       std::dynamic_pointer_cast<vm::DTREagerBlobObject>(JUST(dtr_tensor->eager_blob_object()));
  //   std::stringstream ss;
  //   ss << "tensor: " << dtr_tensor.get() << ", compute op: " <<
  //   dtr_ebo->compute_op_type_name()
  //      << ", is_in_memory: " << dtr_ebo->is_in_memory()
  //      << ", is_evictable: " << dtr_ebo->is_evictable() << ", pinned: " <<
  //      dtr_ebo->num_pinned()
  //      << ", id: " << dtr_ebo->id();
  //   return ss.str();
  // });
  m.def("evict", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    JUST(t->eager_blob_object())->tensor_storage()->Evict(false);
    return Maybe<void>::Ok();
  });
  m.def("is_evictable", [](const std::shared_ptr<one::Tensor>& t) -> Maybe<bool> {
    return JUST(t->eager_blob_object())->tensor_storage()->is_evictable();
  });
  m.def("set_evictable", [](const std::shared_ptr<one::Tensor>& t, bool evictable) -> Maybe<void> {
    JUST(t->eager_blob_object())->tensor_storage()->set_evictable(evictable);
    return Maybe<void>::Ok();
  });
}

}  // namespace oneflow
