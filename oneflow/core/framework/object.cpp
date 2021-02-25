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
#include "oneflow/core/framework/object.h"

namespace oneflow {

namespace compatible_py {

Object::Object(int64_t object_id, const std::shared_ptr<ParallelDesc>& parallel_desc_symbol)
    : object_id_(object_id), parallel_desc_symbol_(parallel_desc_symbol) {}

int64_t Object::object_id() const { return object_id_; }

std::shared_ptr<ParallelDesc> Object::parallel_desc_symbol() const { return parallel_desc_symbol_; }

BlobObject::BlobObject(int64_t object_id,
                       const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr,
                       const std::shared_ptr<OpArgBlobAttribute>& op_arg_blob_attr)
    : Object(object_id, op_arg_parallel_attr->parallel_desc_symbol()),
      op_arg_parallel_attr_(op_arg_parallel_attr),
      op_arg_blob_attr_(op_arg_blob_attr) {}

std::shared_ptr<OpArgParallelAttribute> BlobObject::op_arg_parallel_attr() const {
  return op_arg_parallel_attr_;
}
std::shared_ptr<OpArgBlobAttribute> BlobObject::op_arg_blob_attr() const {
  return op_arg_blob_attr_;
}

void BlobObject::add_releaser(const std::function<void(Object*)>& release) {
  release_.emplace_back(release);
}

void BlobObject::ForceReleaseAll() {
  for (const auto& release : release_) { release(this); }
  release_.clear();
}

}  // namespace compatible_py

}  // namespace oneflow
