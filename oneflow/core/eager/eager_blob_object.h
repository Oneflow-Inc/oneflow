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
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

namespace eager {

class TensorBuffer {
 public:
  char* blob_dptr() { return blob_dptr_.get(); }
  void set_blob_dptr(std::unique_ptr<char, std::function<void(char*)>>&& blob_dptr) {
    blob_dptr_ = std::move(blob_dptr);
  }

 private:
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
};

class EagerBlobObject : public BlobObject {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, const std::shared_ptr<Shape>& shape,
                  DataType data_type, const std::shared_ptr<TensorBuffer>& tensor_buffer)
      : BlobObject(mem_case, shape, data_type), tensor_buffer_(tensor_buffer), blob_body_bytes_(0) {
    CHECK(static_cast<bool>(shape));
    CHECK(static_cast<bool>(tensor_buffer));
  }
  virtual ~EagerBlobObject() override = default;

  virtual BlobDesc* mut_blob_desc() override { return &blob_desc_; }

  virtual const Blob& blob() const override { return *blob_; }
  virtual Blob* mut_blob() override { return blob_.get(); }
  virtual Maybe<void> TryInitBlob() override;

  virtual void TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override;

 private:
  Maybe<void> InitBlob();

  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char, std::function<void(char*)>> header_buffer_;
  std::shared_ptr<TensorBuffer> tensor_buffer_;
  std::size_t blob_body_bytes_;
  MemoryAllocator non_pod_initer_;

 protected:
  std::unique_ptr<RtBlobDesc> rt_blob_desc_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
