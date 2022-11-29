#ifndef ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_
#define ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {
namespace vm {

class OpCallInstructionPolicy;
class DtrOpCallInstructionPolicy;

class TensorStorage {
 public:
  TensorStorage();
  OF_DISALLOW_COPY_AND_MOVE(TensorStorage);

  virtual ~TensorStorage();

  virtual bool is_allocated_in_vm() const = 0;

  size_t blob_bytes() const { return blob_bytes_; }

  char* blob_dptr() { return blob_dptr_.get(); }

  MemoryAllocator* non_pod_allocator() { return non_pod_allocator_.get(); }

  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr, size_t bytes) {
    blob_dptr_ = std::move(blob_dptr);
    blob_bytes_ = bytes;
    is_initialized_ = true;
  }

  const Optional<Symbol<::oneflow::Stream>>& producer_stream() const { return producer_stream_; }
  Maybe<void> init_producer_stream(Symbol<::oneflow::Stream> producer_stream) {
    CHECK_OR_RETURN(!producer_stream_.has_value());
    producer_stream_ = producer_stream;
    return Maybe<void>::Ok();
  }

  const Optional<Symbol<::oneflow::Stream>>& last_used_stream() const { return last_used_stream_; }
  void set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream) {
    last_used_stream_ = last_used_stream;
  }

  void Release() {
    non_pod_allocator_.reset();
    blob_dptr_.reset();
  }

  void RegisterStorageDeleteHook(const std::function<void()>& hook) {
    storage_delete_hooks_.emplace_back(hook);
  }

  void set_compute_op(const OpCallInstructionPolicy& compute_op);
  void clear_compute_op();
  OpCallInstructionPolicy compute_op() const;
  void Evict(bool eager_eviction);
  void Pin() { num_pinned_++; }
  void Unpin() { num_pinned_--; }
  void Access();
  bool is_in_memory() const { return blob_bytes_ == 0 || blob_dptr_ != nullptr; }
  bool is_pinned() const { return num_pinned() > 0; }
  int32_t num_pinned() const { return num_pinned_; }
  bool is_evictable() const { return compute_op_ != nullptr && !eviction_disabled_; }
  void disable_eviction() { eviction_disabled_ = true; }
  bool eviction_disabled() const { return eviction_disabled_; }
  int64_t id() const { return id_; }
  Maybe<double> cost(size_t override_size) const;
  std::string compute_op_type_name() const;
  bool is_initialized() const { return is_initialized_; }
  bool set_initialized() { return is_initialized_ = true; }

 private:
  bool is_initialized_ = false;
  int64_t id_;
  size_t num_pinned_;
  size_t blob_bytes_;
  bool eviction_disabled_ = false;
  double last_access_time_;
  double compute_time_;
  std::shared_ptr<DtrOpCallInstructionPolicy> compute_op_;

  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<::oneflow::Stream>> producer_stream_;
  Optional<Symbol<::oneflow::Stream>> last_used_stream_;
  std::vector<std::function<void()>> storage_delete_hooks_;
};

class InsideVmTensorStorage : public TensorStorage {
 public:
  InsideVmTensorStorage() = default;
  ~InsideVmTensorStorage() = default;

  bool is_allocated_in_vm() const override { return true; }
};

class OutsideVmTensorStorage : public TensorStorage {
 public:
  OutsideVmTensorStorage() = default;
  ~OutsideVmTensorStorage() = default;

  bool is_allocated_in_vm() const override { return false; }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_
