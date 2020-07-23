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
#ifndef ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace eager {

class BlobObject : public vm::Object {
 public:
  BlobObject(const BlobObject&) = delete;
  BlobObject(BlobObject&&) = delete;
  BlobObject(const std::shared_ptr<MemoryCase>& mem_case, DataType data_type)
      : mem_case_(mem_case), blob_desc_(data_type), blob_body_bytes_(0) {}
  ~BlobObject() override = default;

  const BlobDesc& blob_desc() const { return blob_desc_; }
  BlobDesc* mut_blob_desc() { return &blob_desc_; }

  const Blob& blob() const { return *blob_; }
  Blob* mut_blob() { return blob_.get(); }
  Maybe<void> TryInitBlob();

  void TryAllocateBlobBodyMemory(DeviceCtx* device_ctx);

 private:
  Maybe<void> InitBlob();

  std::shared_ptr<MemoryCase> mem_case_;
  BlobDesc blob_desc_;
  std::unique_ptr<RtBlobDesc> rt_blob_desc_;
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<char, std::function<void(char*)>> header_buffer_;
  std::unique_ptr<char, std::function<void(char*)>> blob_dptr_;
  std::size_t blob_body_bytes_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
