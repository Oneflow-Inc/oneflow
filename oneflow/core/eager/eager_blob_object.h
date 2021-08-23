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

namespace oneflow {

namespace vm {

class TensorBuffer {
 public:
  char* blob_dptr() { return blob_dptr_.get(); }
  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr) {
    blob_dptr_ = std::move(blob_dptr);
  }

  void reset() { blob_dptr_.reset(); }

 private:
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
};

class EagerBlobObject final : public BlobObject {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer)
      : EagerBlobObject(mem_case, shape, data_type, tensor_buffer, Optional<LocalDepObject*>()) {}

  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer,
                  LocalDepObject* dep_object)
      : EagerBlobObject(mem_case, shape, data_type, tensor_buffer,
                        Optional<LocalDepObject*>(dep_object)) {}

  ~EagerBlobObject() override {
    non_pod_initer_.reset();
    tensor_buffer_.reset();
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
    non_pod_initer_.reset();
    tensor_buffer_->reset();
    return Maybe<void>::Ok();
  }

  Maybe<LocalDepObject*> compute_local_dep_object() const {
    return compute_local_dep_object_.value();
  }

  std::shared_ptr<TensorBuffer>& tensor_buffer() { return tensor_buffer_; }

  bool is_shape_synced() const { return is_shape_synced_; }

  void set_is_shape_synced(bool val) { is_shape_synced_ = val; }

 private:
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer,
                  const Optional<LocalDepObject*>& dep_object);

  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char[]> header_buffer_;
  std::shared_ptr<TensorBuffer> tensor_buffer_;
  std::size_t blob_body_bytes_;
  std::unique_ptr<MemoryAllocator> non_pod_initer_;
  std::atomic<bool> is_shape_synced_;
  Optional<LocalDepObject*> compute_local_dep_object_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
