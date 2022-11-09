#ifndef ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_
#define ONEFLOW_CORE_EAGER_TENSOR_STORAGE_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {
namespace vm {

class OpCallInstructionPolicy;

class TensorStorage {
 public:
  TensorStorage()
      : blob_bytes_(0),
        non_pod_allocator_(std::make_unique<MemoryAllocator>()),
        producer_stream_(NullOpt),
        last_used_stream_(NullOpt) {}
  OF_DISALLOW_COPY_AND_MOVE(TensorStorage);

  virtual ~TensorStorage() {
    for (const auto& hook : storage_delete_hooks_) { hook(); }
  }

  virtual bool is_allocated_in_vm() const = 0;

  size_t blob_bytes() const { return blob_bytes_; }

  char* blob_dptr() { return blob_dptr_.get(); }

  MemoryAllocator* non_pod_allocator() { return non_pod_allocator_.get(); }

  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr, size_t bytes) {
    blob_dptr_ = std::move(blob_dptr);
    blob_bytes_ = bytes;
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

  void set_compute_op(OpCallInstructionPolicy* compute_op);
  OpCallInstructionPolicy* compute_op() const { return compute_op_.get(); }

  void Evict(bool eager_eviction) { return Release(); }
  void Pin() { UNIMPLEMENTED(); }
  bool is_pinned() const { return num_pinned() > 0; }
  int32_t num_pinned() const { UNIMPLEMENTED(); }
  bool is_evictable() const { UNIMPLEMENTED(); }

  size_t id() const { UNIMPLEMENTED(); }
  Maybe<double> cost(size_t override_size) const { UNIMPLEMENTED(); }
  std::string compute_op_type_name() const { UNIMPLEMENTED(); }

 private:
  size_t blob_bytes_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<::oneflow::Stream>> producer_stream_;
  Optional<Symbol<::oneflow::Stream>> last_used_stream_;
  std::vector<std::function<void()>> storage_delete_hooks_;
  std::shared_ptr<OpCallInstructionPolicy> compute_op_;
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
