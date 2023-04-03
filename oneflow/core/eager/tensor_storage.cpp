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
#include "oneflow/core/eager/tensor_storage.h"

#include "oneflow/core/vm/virtual_machine.h"

namespace oneflow {
namespace vm {

TensorStorage::TensorStorage(bool is_allocated_in_vm)
    : blob_bytes_(0),
      non_pod_allocator_(std::make_unique<MemoryAllocator>()),
      producer_stream_(NullOpt),
      last_used_stream_(NullOpt),
      is_allocated_in_vm_(is_allocated_in_vm) {}

TensorStorage::~TensorStorage() {
  for (const auto& hook : storage_delete_hooks_) { hook(); }
}

void TensorStorage::Release() {
  non_pod_allocator_.reset();
  blob_dptr_.reset();
}

Maybe<void> TensorStorage::init_producer_stream(Symbol<::oneflow::Stream> producer_stream) {
  CHECK_OR_RETURN(!producer_stream_.has_value());
  producer_stream_ = producer_stream;
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
