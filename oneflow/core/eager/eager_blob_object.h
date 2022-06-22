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

  const Optional<Symbol<Stream>>& producer_stream() const { return producer_stream_; }
  Maybe<void> init_producer_stream(Symbol<Stream> producer_stream) {
    CHECK_OR_RETURN(!producer_stream_.has_value());
    producer_stream_ = producer_stream;
    return Maybe<void>::Ok();
  }

  const Optional<Symbol<Stream>>& last_used_stream() const { return last_used_stream_; }
  void set_last_used_stream(Symbol<Stream> last_used_stream) {
    last_used_stream_ = last_used_stream;
  }

  void Release() {
    non_pod_allocator_.reset(new MemoryAllocator());
    blob_dptr_.reset();
  }

  void RegisterStorageDeleteHook(const std::function<void()>& hook) {
    storage_delete_hooks_.emplace_back(hook);
  }

 private:
  size_t blob_bytes_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<Stream>> producer_stream_;
  Optional<Symbol<Stream>> last_used_stream_;
  std::vector<std::function<void()>> storage_delete_hooks_;
};

class EagerBlobObject {
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

  virtual ~EagerBlobObject() {
    tensor_storage_.reset();
    blob_.reset();
  }

  std::vector<float> backup_data_;
  float hash_ = -1;

  BlobDesc* mut_blob_desc() { return &blob_desc_; }

  void set_storage_offset(const int64_t offset);

  [[deprecated("\"Blob\" will be removed in eager. Please avoid to use this method whenever "
               "possible. Almost all methods of `Blob` are also in `EagerBlobObject`.")]] Blob*
  blob();

  virtual Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx);
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

  const Optional<Symbol<Stream>>& producer_stream() const {
    return tensor_storage_->producer_stream();
  }
  Maybe<void> init_producer_stream(Symbol<Stream> producer_stream) {
    return tensor_storage_->init_producer_stream(producer_stream);
  }

  const Optional<Symbol<Stream>>& last_used_stream() const {
    return tensor_storage_->last_used_stream();
  }
  void set_last_used_stream(Symbol<Stream> last_used_stream) {
    tensor_storage_->set_last_used_stream(last_used_stream);
  }

  void set_pin_memory(const bool pin_memory) { pin_memory_ = pin_memory; }

  bool pin_memory() const { return pin_memory_; }

  std::shared_ptr<const Shape> shape_ptr() const { return shape_; }
  const Shape& shape() const { return *shape_; }
  Shape& mut_shape() { return *shape_; }
  std::shared_ptr<const Stride> stride_ptr() const { return stride_; }
  const Stride& stride() const { return *stride_; }
  Stride& mut_stride() { return *stride_; }

  size_t ByteSizeOfBlobBody() const { return shape_->elem_cnt() * GetSizeOfDataType(data_type_); }
  size_t AlignedByteSizeOfBlobBody() const {
    return RoundUp(ByteSizeOfBlobBody(), kBlobBodyAlignSize);
  }
  size_t ByteSizeOfBlobHeader() const { return shape().NumAxes() * sizeof(int64_t); }
  size_t AlignedByteSizeOfBlobHeader() const {
    return RoundUp(ByteSizeOfBlobHeader(), kBlobHeaderAlignSize);
  }

  template<typename T = void>
  const T* dptr() const {
    return reinterpret_cast<T*>(tensor_storage_->blob_dptr()
                                + storage_offset_ * GetSizeOfDataType(data_type_));
  }

  template<typename T = void>
  T* mut_dptr() {
    return const_cast<T*>(dptr<T>());
  }

  const char* header_ptr() const { return reinterpret_cast<const char*>(shape_->dim_vec().data()); }
  char* mut_header_ptr() { return reinterpret_cast<char*>(shape_->dim_vec().data()); }

  DataType data_type() const { return data_type_; }
  DataType* mut_data_type() { return &data_type_; }
  const MemoryCase& mem_case() const { return *mem_case_; }

  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }
  bool* mut_is_dynamic() { return &is_dynamic_; }

 private:
  bool is_dynamic_;
  std::shared_ptr<MemoryCase> mem_case_;
  DataType data_type_;
  std::shared_ptr<Shape> shape_;
  std::shared_ptr<Stride> stride_;
  int64_t storage_offset_;
 protected:
  std::shared_ptr<TensorStorage> tensor_storage_;
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
