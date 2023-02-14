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
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/framework/shut_down_util.h"

namespace oneflow {
namespace one {

TensorStorage::TensorStorage(const std::shared_ptr<const ParallelDesc>& parallel_desc)
    : storage_(std::make_shared<vm::TensorStorage>(true)) {}

TensorStorage::TensorStorage(const std::shared_ptr<vm::TensorStorage>& tensor_storage)
    : storage_(tensor_storage) {}

TensorStorage::~TensorStorage() {
  if (!IsShuttingDown() && releaser_hook_) { (*releaser_hook_)(storage_); }
}

}  // namespace one
}  // namespace oneflow
