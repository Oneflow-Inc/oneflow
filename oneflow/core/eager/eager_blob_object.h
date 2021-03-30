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

class EagerBlobObject : public BlobObject {
 public:
  EagerBlobObject(const EagerBlobObject&) = delete;
  EagerBlobObject(EagerBlobObject&&) = delete;
  EagerBlobObject(const std::shared_ptr<MemoryCase>& mem_case, DataType data_type)
      : BlobObject(mem_case, data_type), blob_body_bytes_(0) {}
  virtual ~EagerBlobObject() override = default;

  virtual BlobDesc* mut_blob_desc() override { return &blob_desc_; }

  virtual const Blob& blob() const override { return *blob_; }
  virtual Blob* mut_blob() override { return blob_.get(); }
  virtual Maybe<void> TryInitBlob() override;
  Maybe<void> InitBlob();

  virtual void TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override;

 private:

  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char, std::function<void(char*)>> header_buffer_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::size_t blob_body_bytes_;
  MemoryAllocator non_pod_initer_;

 protected:
  std::unique_ptr<RtBlobDesc> rt_blob_desc_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_BLOB_OBJECT_H_
