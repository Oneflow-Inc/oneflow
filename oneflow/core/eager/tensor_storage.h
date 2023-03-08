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
#ifndef ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_
#define ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {

namespace vm {

class TensorStorage {
 public:
  explicit TensorStorage(bool is_allocated_in_vm);
  OF_DISALLOW_COPY_AND_MOVE(TensorStorage);

  virtual ~TensorStorage();

  bool is_allocated_in_vm() const { return is_allocated_in_vm_; }

  size_t blob_bytes() const { return blob_bytes_; }

  char* blob_dptr() { return blob_dptr_.get(); }

  MemoryAllocator* non_pod_allocator() { return non_pod_allocator_.get(); }

  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr, size_t bytes) {
    blob_dptr_ = std::move(blob_dptr);
    blob_bytes_ = bytes;
  }

  const Optional<Symbol<::oneflow::Stream>>& producer_stream() const { return producer_stream_; }
  Maybe<void> init_producer_stream(Symbol<::oneflow::Stream> producer_stream);

  const Optional<Symbol<::oneflow::Stream>>& last_used_stream() const { return last_used_stream_; }
  void set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream) {
    last_used_stream_ = last_used_stream;
  }

  virtual void Release();

  void RegisterStorageDeleteHook(const std::function<void()>& hook) {
    storage_delete_hooks_.emplace_back(hook);
  }
  Symbol<Device> device() const;

 protected:
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  size_t blob_bytes_;

 private:
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<::oneflow::Stream>> producer_stream_;
  Optional<Symbol<::oneflow::Stream>> last_used_stream_;
  std::vector<std::function<void()>> storage_delete_hooks_;
  bool is_allocated_in_vm_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_
