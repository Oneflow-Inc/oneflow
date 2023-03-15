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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"

namespace oneflow {

CnnlWorkspace::CnnlWorkspace(ep::MluStream* stream, size_t workspace_size)
    : mlu_stream_(stream),
      size_(workspace_size),
      capacity_(workspace_size),
      workspace_dptr_(nullptr) {
  if (capacity_ > 0) {
    auto* allocator = mlu_stream_->workspace_allocator();
    CHECK_JUST(allocator->Allocate(&workspace_dptr_, capacity_));
  }
}

CnnlWorkspace::~CnnlWorkspace() {
  if (capacity_ > 0 && !workspace_dptr_) {
    auto* allocator = mlu_stream_->workspace_allocator();
    allocator->Deallocate(workspace_dptr_, capacity_);
  }
  workspace_dptr_ = nullptr;
}

void CnnlWorkspace::resize(size_t workspace_size) {
  if (capacity_ < workspace_size) {
    auto* allocator = mlu_stream_->workspace_allocator();
    allocator->Deallocate(workspace_dptr_, capacity_);
    capacity_ = workspace_size;
    CHECK_JUST(allocator->Allocate(&workspace_dptr_, capacity_));
  }
  size_ = workspace_size;
}

}  // namespace oneflow
