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
#ifndef ONEFLOW_CORE_VM_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_ALLOCATOR_H_

#include <cstddef>
#include "oneflow/core/common/maybe.h"
#include "glog/logging.h"

namespace oneflow {
namespace vm {

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual Maybe<void> Allocate(char** mem_ptr, std::size_t size) = 0;
  virtual void Deallocate(char* mem_ptr, std::size_t size) = 0;
  virtual void DeviceReset() = 0;

 protected:
  Allocator() = default;
};

class UnimplementedAllocator final : public Allocator {
 public:
  explicit UnimplementedAllocator(const std::string& debug_str) : debug_str_(debug_str) {}
  virtual ~UnimplementedAllocator() = default;

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override {
    UNIMPLEMENTED_THEN_RETURN() << debug_str_;
  }

  void Deallocate(char* mem_ptr, std::size_t size) override { LOG(FATAL) << debug_str_; }
  void DeviceReset() override { LOG(FATAL) << debug_str_; }

 private:
  std::string debug_str_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ALLOCATOR_H_
