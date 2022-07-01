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
#ifndef ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace vm {

class TensorStorage {
 public:
  TensorStorage()
      : non_pod_allocator_(std::make_unique<MemoryAllocator>()),
        producer_stream_(NullOpt),
        last_used_stream_(NullOpt) {}

  ~TensorStorage() {
    for (const auto& hook : storage_delete_hooks_) { hook(); }
  }

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

 private:
  size_t blob_bytes_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<::oneflow::Stream>> producer_stream_;
  Optional<Symbol<::oneflow::Stream>> last_used_stream_;
  std::vector<std::function<void()>> storage_delete_hooks_;
};

class EagerBlobObject final : public user_op::Tensor,
                              public user_op::TensorDesc,
                              public std::enable_shared_from_this<EagerBlobObject> {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  const std::shared_ptr<Stride>& stride, DataType data_type,
                  const std::shared_ptr<TensorStorage>& tensor_storage)
      : EagerBlobObject(mem_case, shape, stride, data_type, tensor_storage,
                        intrusive::shared_ptr<LocalDepObject>()) {}
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  const std::shared_ptr<Stride>& stride, DataType data_type,
                  const std::shared_ptr<TensorStorage>& tensor_storage,
                  const intrusive::shared_ptr<LocalDepObject>& dep_object);

  ~EagerBlobObject() { tensor_storage_.reset(); }

  // user_op::TensorDesc overrides
  const Shape& shape() const override { return *shape_; }
  Shape* mut_shape() override { return shape_.get(); }
  const Stride& stride() const override { return *stride_; }
  Stride* mut_stride() override { return stride_.get(); }
  DataType data_type() const override { return data_type_; }
  DataType* mut_data_type() override { return &data_type_; }
  bool is_dynamic() const override { return is_dynamic_; }
  bool* mut_is_dynamic() override { return &is_dynamic_; }
  void set_is_dynamic(bool is_dynamic) override { is_dynamic_ = is_dynamic; }

  // user_op::Tensor overrides
  ShapeView shape_view() const override { return *shape_; }
  MutShapeView mut_shape_view() override { return *shape_; }
  const MemoryCase& mem_case() const override { return *mem_case_; }
  const void* raw_dptr() const override {
    CHECK(inited_mem_ptr_for_allocation_compuation_pipelining_)
        << "mem_ptr_for_allocation_compuation_pipelining_ not initialized. Please check if there "
           "are any EagerBlobObjects created outside vm";
    return mem_ptr_for_allocation_compuation_pipelining_
           + storage_offset_ * GetSizeOfDataType(data_type_);
  }
  void* mut_raw_dptr() override { return const_cast<void*>(raw_dptr()); }

  void set_storage_offset(const int64_t offset);

  [[deprecated("\"Blob\" will be removed in eager. Please avoid to use this method whenever "
               "possible. Almost all methods of `Blob` are also in `EagerBlobObject`.")]] Blob*
  blob();

  Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx);
  Maybe<void> DeallocateBlobDataPtr() {
    tensor_storage_->Release();
    tensor_storage_.reset(new TensorStorage);
    return Maybe<void>::Ok();
  }
  void RegisterStorageDeleteHook(const std::function<void()>& hook) {
    tensor_storage_->RegisterStorageDeleteHook(hook);
  }

  Maybe<LocalDepObject*> compute_local_dep_object() const {
    CHECK_NOTNULL_OR_RETURN(compute_local_dep_object_.get());
    return compute_local_dep_object_.get();
  }

  std::shared_ptr<TensorStorage>& tensor_storage() { return tensor_storage_; }

  bool is_shape_synced() const { return is_shape_synced_; }

  void set_is_shape_synced(bool val) { is_shape_synced_ = val; }

  const Optional<Symbol<::oneflow::Stream>>& producer_stream() const {
    return tensor_storage_->producer_stream();
  }
  Maybe<void> init_producer_stream(Symbol<::oneflow::Stream> producer_stream) {
    return tensor_storage_->init_producer_stream(producer_stream);
  }

  const Optional<Symbol<::oneflow::Stream>>& last_used_stream() const {
    return tensor_storage_->last_used_stream();
  }
  void set_last_used_stream(Symbol<::oneflow::Stream> last_used_stream) {
    tensor_storage_->set_last_used_stream(last_used_stream);
  }

  std::shared_ptr<const Shape> shape_ptr() const { return shape_; }
  std::shared_ptr<const Stride> stride_ptr() const { return stride_; }

  size_t ByteSizeOfBlobBody() const { return shape_->elem_cnt() * GetSizeOfDataType(data_type_); }
  size_t AlignedByteSizeOfBlobBody() const {
    return RoundUp(ByteSizeOfBlobBody(), kBlobBodyAlignSize);
  }
  size_t ByteSizeOfBlobHeader() const { return shape().NumAxes() * sizeof(int64_t); }
  size_t AlignedByteSizeOfBlobHeader() const {
    return RoundUp(ByteSizeOfBlobHeader(), kBlobHeaderAlignSize);
  }

  const char* header_ptr() const { return reinterpret_cast<const char*>(shape_->dim_vec().data()); }
  char* mut_header_ptr() { return reinterpret_cast<char*>(shape_->dim_vec().data()); }

  void InitOrCheckMemPtrForAllocationComputationPipelining() {
    auto* ptr = tensor_storage_->blob_dptr();
    if (inited_mem_ptr_for_allocation_compuation_pipelining_) {
      CHECK_EQ(mem_ptr_for_allocation_compuation_pipelining_, ptr);
    } else {
      mem_ptr_for_allocation_compuation_pipelining_ = ptr;
      inited_mem_ptr_for_allocation_compuation_pipelining_ = true;
    }
  }

 private:
  void InitMemPtrForAllocationComputationPipelining() {
    auto* ptr = tensor_storage_->blob_dptr();
    CHECK(!inited_mem_ptr_for_allocation_compuation_pipelining_)
        << "mem_ptr_for_allocation_compuation_pipelining_ has been initialized.";
    mem_ptr_for_allocation_compuation_pipelining_ = ptr;
    inited_mem_ptr_for_allocation_compuation_pipelining_ = true;
  }

  bool is_dynamic_;
  std::shared_ptr<MemoryCase> mem_case_;
  DataType data_type_;
  std::shared_ptr<Shape> shape_;
  std::shared_ptr<Stride> stride_;
  int64_t storage_offset_;
  std::shared_ptr<TensorStorage> tensor_storage_;
  // For allocation-computation pipeline, the value of mem_ptr_for_allocation_compuation_pipelining_
  // are kept even after tensor_storage_.reset().
  char* mem_ptr_for_allocation_compuation_pipelining_;
  bool inited_mem_ptr_for_allocation_compuation_pipelining_;
  std::atomic<bool> is_shape_synced_;
  bool pin_memory_;
  intrusive::shared_ptr<LocalDepObject> compute_local_dep_object_;

  // NOTE: Will be removed soon. Avoid to use it whenever possible.
  BlobDesc blob_desc_;
  std::unique_ptr<Blob> blob_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
