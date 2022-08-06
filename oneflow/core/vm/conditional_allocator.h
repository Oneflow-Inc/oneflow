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
#ifndef ONEFLOW_CORE_VM_CONDITIONAL_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_CONDITIONAL_ALLOCATOR_H_

#include <functional>
#include <memory>

#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/allocator.h"

namespace oneflow {
namespace vm {

class ConditionalAllocator final : public Allocator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConditionalAllocator);
  ConditionalAllocator(std::function<bool()> condition, std::unique_ptr<Allocator> allocator1,
                       std::unique_ptr<Allocator> allocator2);
  ~ConditionalAllocator() override = default;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;

  Allocator* get_allocator();

 private:
  std::function<bool()> condition_;
  std::unique_ptr<Allocator> allocator1_;
  std::unique_ptr<Allocator> allocator2_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONDITIONAL_ALLOCATOR_H_
