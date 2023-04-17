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
namespace remat {
class DisjNode;
}

namespace vm {

class OpCallInstructionPolicy;
class DtrOpCallInstructionPolicy;

class TensorStorageBase {
 public:
  explicit TensorStorageBase(bool is_allocated_in_vm, Symbol<Device> device);
  OF_DISALLOW_COPY_AND_MOVE(TensorStorageBase);

  virtual ~TensorStorageBase();

  bool is_allocated_in_vm() const { return is_allocated_in_vm_; }

  size_t blob_bytes() const { return blob_bytes_; }

  char* blob_dptr() { return blob_dptr_.get(); }

  MemoryAllocator* non_pod_allocator() { return non_pod_allocator_.get(); }

  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr, size_t bytes) {
    blob_dptr_ = std::move(blob_dptr);
    blob_bytes_ = bytes;
    is_initialized_ = true;
  }

  const Optional<Symbol<::oneflow::Stream>>& producer_stream() const { return producer_stream_; }
  Maybe<void> init_producer_stream(Symbol<::oneflow::Stream> producer_stream);

  const Optional<Symbol<::oneflow::Stream>>& last_used_stream() const { return last_used_stream_; }
  void set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream) {
    last_used_stream_ = last_used_stream;
  }

  void _Release();
  virtual void Release();

  void RegisterStorageDeleteHook(const std::function<void()>& hook) {
    storage_delete_hooks_.emplace_back(hook);
  }
  Symbol<Device> device() const;

 protected:
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  size_t blob_bytes_;
  bool is_initialized_ = false;
  Symbol<Device> device_;

 private:
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<::oneflow::Stream>> producer_stream_;
  Optional<Symbol<::oneflow::Stream>> last_used_stream_;
  std::vector<std::function<void()>> storage_delete_hooks_;
  bool is_allocated_in_vm_;
};

class RematableTensorStorage final : public TensorStorageBase {
 public:
  explicit RematableTensorStorage(Symbol<Device> device);
  OF_DISALLOW_COPY_AND_MOVE(RematableTensorStorage);
  ~RematableTensorStorage() override;

  void set_compute_op(const std::shared_ptr<DtrOpCallInstructionPolicy>& compute_op,
                      double compute_time);
  void clear_compute_op();
  OpCallInstructionPolicy compute_op() const;
  std::shared_ptr<DtrOpCallInstructionPolicy> dtr_compute_op() const;
  void Release() override;
  void Remat();
  void Evict(bool eager_eviction);
  void Pin();
  void Unpin();
  void Access();
  bool is_in_memory() const { return blob_bytes_ == 0 || blob_dptr_ != nullptr; }
  bool is_pinned() const { return num_pinned() > 0; }
  int32_t num_pinned() const { return num_pinned_; }
  bool is_evictable() const;
  void set_eviction_disabled(bool disabled) { eviction_disabled_ = disabled; }
  bool is_eviction_disabled() const { return eviction_disabled_; }
  int64_t id() const { return id_; }
  Maybe<double> cost(size_t override_size) const;
  double approx_neighbor_cost() const;
  std::string compute_op_type_name() const;
  bool is_initialized() const { return is_initialized_; }
  void set_initialized() { is_initialized_ = true; }
  bool is_needed_by_backward() const { return is_needed_by_backward_; }
  void set_needed_by_backward() { is_needed_by_backward_ = true; }
  double compute_time() const { return compute_time_; }
  std::shared_ptr<remat::DisjNode> node;

 private:
  int64_t id_{};
  size_t num_pinned_{};
  bool eviction_disabled_ = false;
  double last_access_time_{};
  double compute_time_{};
  std::shared_ptr<DtrOpCallInstructionPolicy> compute_op_;
  bool is_needed_by_backward_ = false;

  void LogEviction(bool eager_eviction) const;
};

class TensorStorage : public TensorStorageBase {
 public:
  explicit TensorStorage(const std::shared_ptr<TensorStorage>& tensor_storage)
      : TensorStorageBase(tensor_storage->is_allocated_in_vm(), tensor_storage->device()) {}

  ~TensorStorage() override = default;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_
