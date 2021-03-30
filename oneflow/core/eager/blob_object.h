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

namespace oneflow {

class ParallelDesc;

namespace eager {

class BlobObject : public vm::Object {
 public:
  BlobObject(const std::shared_ptr<MemoryCase>& mem_case, DataType data_type)
      : mem_case_(mem_case), blob_desc_(data_type) {}
  BlobObject(const BlobObject&) = delete;
  BlobObject(BlobObject&&) = delete;
  virtual ~BlobObject() override = default;

  const BlobDesc& blob_desc() const { return blob_desc_; }
  virtual BlobDesc* mut_blob_desc() = 0;

  virtual const Blob& blob() const = 0;
  virtual Blob* mut_blob() = 0;
  virtual Maybe<void> TryInitBlob() = 0;
  virtual Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) = 0;
  virtual Maybe<void> DeallocateBlobDataPtr() = 0;

  Maybe<void> CheckMemCase(const ParallelDesc& parallel_desc, int64_t machine_id) const;

  const MemoryCase& mem_case() const { return *mem_case_; }

 protected:
  std::shared_ptr<MemoryCase> mem_case_;
  BlobDesc blob_desc_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_BLOB_OBJECT_H_
