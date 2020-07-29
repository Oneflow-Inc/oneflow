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
#ifndef ONEFLOW_CORE_VM_MEM_BUFFER_OBJECT_H_
#define ONEFLOW_CORE_VM_MEM_BUFFER_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {
namespace vm {

class MemBufferObjectType final : public Object {
 public:
  MemBufferObjectType(const MemBufferObjectType&) = delete;
  MemBufferObjectType(MemBufferObjectType&) = delete;

  MemBufferObjectType() = default;
  ~MemBufferObjectType() override = default;

  const MemoryCase& mem_case() const { return mem_case_; }
  std::size_t size() const { return size_; }

  MemoryCase* mut_mem_case() { return &mem_case_; }
  void set_size(std::size_t val) { size_ = val; }

 private:
  MemoryCase mem_case_;
  std::size_t size_;
};

class MemBufferObjectValue final : public Object {
 public:
  MemBufferObjectValue(const MemBufferObjectValue&) = delete;
  MemBufferObjectValue(MemBufferObjectValue&) = delete;

  MemBufferObjectValue() = default;
  ~MemBufferObjectValue() override = default;

  const char* data() const { return data_; }
  char* mut_data() { return data_; }
  void reset_data(char* val) { data_ = val; }

 private:
  char* data_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_BUFFER_OBJECT_H_
