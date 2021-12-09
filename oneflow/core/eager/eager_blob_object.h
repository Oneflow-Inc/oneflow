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
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor_methods.h"

namespace oneflow {

namespace vm {

class TensorStorage {
 public:
  TensorStorage()
      : non_pod_allocator_(std::make_unique<MemoryAllocator>()),
        producer_op_device_(NullOpt),
        last_used_device_(NullOpt) {}

  size_t blob_bytes() const { return blob_bytes_; }

  char* blob_dptr() { return blob_dptr_.get(); }

  MemoryAllocator* non_pod_allocator() { return non_pod_allocator_.get(); }

  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr, size_t bytes) {
    blob_dptr_ = std::move(blob_dptr);
    blob_bytes_ = bytes;
  }

  const Optional<Symbol<Device>>& producer_op_device() const { return producer_op_device_; }
  Maybe<void> init_producer_op_device(Symbol<Device> producer_op_device) {
    CHECK_OR_RETURN(!producer_op_device_.has_value());
    producer_op_device_ = producer_op_device;
    return Maybe<void>::Ok();
  }

  const Optional<Symbol<Device>>& last_used_device() const { return last_used_device_; }
  void set_last_used_device(Symbol<Device> last_used_device) {
    last_used_device_ = last_used_device;
  }

 private:
  size_t blob_bytes_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::unique_ptr<MemoryAllocator> non_pod_allocator_;
  Optional<Symbol<Device>> producer_op_device_;
  Optional<Symbol<Device>> last_used_device_;
};

class EagerBlobObject final : public BlobObject {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorStorage>& tensor_storage)
      : EagerBlobObject(mem_case, shape, data_type, tensor_storage, Optional<LocalDepObject*>()) {}

  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorStorage>& tensor_storage,
                  LocalDepObject* dep_object)
      : EagerBlobObject(mem_case, shape, data_type, tensor_storage,
                        Optional<LocalDepObject*>(dep_object)) {}

  ~EagerBlobObject() override {
    tensor_storage_.reset();
    header_buffer_.reset();
    blob_.reset();
  }

  BlobDesc* mut_blob_desc() override { return &blob_desc_; }

  const Blob& blob() const override { return *blob_; }
  Blob* mut_blob() override { return blob_.get(); }

  Maybe<void> TryInitBlob() override;
  Maybe<void> InitBlob();

  Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override;
  Maybe<void> DeallocateBlobDataPtr() override {
    tensor_storage_.reset(new TensorStorage);
    return Maybe<void>::Ok();
  }

  Maybe<LocalDepObject*> compute_local_dep_object() const {
    return JUST(compute_local_dep_object_);
  }

  std::shared_ptr<TensorStorage>& tensor_storage() { return tensor_storage_; }

  bool is_shape_synced() const { return is_shape_synced_; }

  void set_is_shape_synced(bool val) { is_shape_synced_ = val; }

  int64_t storage_offset() const { return storage_offset_; }

  void set_storage_offset(int64_t storage_offset) { storage_offset_ = storage_offset; }

  const Optional<Symbol<Device>>& producer_op_device() const {
    return tensor_storage_->producer_op_device();
  }
  Maybe<void> init_producer_op_device(Symbol<Device> producer_op_device) {
    return tensor_storage_->init_producer_op_device(producer_op_device);
  }

  const Optional<Symbol<Device>>& last_used_device() const {
    return tensor_storage_->last_used_device();
  }
  void set_last_used_device(Symbol<Device> last_used_device) {
    tensor_storage_->set_last_used_device(last_used_device);
  }

 private:
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorStorage>& tensor_storage,
                  const Optional<LocalDepObject*>& dep_object);
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char[]> header_buffer_;
  std::shared_ptr<TensorStorage> tensor_storage_;
  std::atomic<bool> is_shape_synced_;
  int64_t storage_offset_;
  Optional<LocalDepObject*> compute_local_dep_object_;
  Optional<Symbol<Device>> producer_op_device_;
  Optional<Symbol<Device>> last_used_device_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
