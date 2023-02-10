#include "oneflow/core/eager/tensor_storage.h"

#include "oneflow/core/vm/virtual_machine.h"

namespace oneflow {
namespace vm {

TensorStorage::TensorStorage(bool is_allocated_in_vm)
    : blob_bytes_(0),
      non_pod_allocator_(std::make_unique<MemoryAllocator>()),
      producer_stream_(NullOpt),
      last_used_stream_(NullOpt),
      is_allocated_in_vm_(is_allocated_in_vm) {}

void TensorStorage::Release() {
  for (const auto& hook : storage_delete_hooks_) { hook(); }
  non_pod_allocator_.reset();
  blob_dptr_.reset();
}

Maybe<void> TensorStorage::init_producer_stream(Symbol<::oneflow::Stream> producer_stream) {
  CHECK_OR_RETURN(!producer_stream_.has_value());
  producer_stream_ = producer_stream;
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow

