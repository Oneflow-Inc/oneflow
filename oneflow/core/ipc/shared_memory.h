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
#ifndef ONEFLOW_CORE_IPC_SHARED_MEMORY_H_
#define ONEFLOW_CORE_IPC_SHARED_MEMORY_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace ipc {

class SharedMemory final {
 public:
  SharedMemory(const SharedMemory&) = delete;
  SharedMemory(SharedMemory&&) = delete;
  ~SharedMemory();

  static Maybe<SharedMemory> Open(size_t size);
  static Maybe<SharedMemory> Open(const std::string& name);

  const char* buf() const { return buf_; }
  char* mut_buf() { return buf_; }

  const std::string& name() const { return name_; }
  size_t size() const { return size_; }

  Maybe<void> Close();
  Maybe<void> Unlink();

 private:
  SharedMemory(char* buf, const std::string& name, size_t size)
      : buf_(buf), name_(name), size_(size) {}

  char* buf_;
  std::string name_;
  size_t size_;
};

}  // namespace ipc
}  // namespace oneflow

#endif  // ONEFLOW_CORE_IPC_SHARED_MEMORY_H_
