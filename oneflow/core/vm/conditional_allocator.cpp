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
#include "oneflow/core/vm/conditional_allocator.h"

#include <cstddef>
#include <utility>

namespace oneflow {
namespace vm {

ConditionalAllocator::ConditionalAllocator(std::function<bool()> condition,
                                           std::unique_ptr<Allocator> allocator1,
                                           std::unique_ptr<Allocator> allocator2)
    : condition_(std::move(condition)),
      allocator1_(std::move(allocator1)),
      allocator2_(std::move(allocator2)) {}

void ConditionalAllocator::Allocate(char** mem_ptr, std::size_t size) {
  if (condition_()) {
    allocator1_->Allocate(mem_ptr, size);
  } else {
    allocator2_->Allocate(mem_ptr, size);
  }
}

void ConditionalAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  if (condition_()) {
    allocator1_->Deallocate(mem_ptr, size);
  } else {
    allocator2_->Deallocate(mem_ptr, size);
  }
}

Allocator* ConditionalAllocator::get_allocator() {
  if (condition_()) {
    return allocator1_.get();
  } else {
    return allocator2_.get();
  }
}

}  // namespace vm
}  // namespace oneflow
