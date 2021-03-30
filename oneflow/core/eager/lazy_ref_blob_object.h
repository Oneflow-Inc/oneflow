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
#ifndef ONEFLOW_CORE_EAGER_LAZY_REF_BLOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_LAZY_REF_BLOB_OBJECT_H_

#include "oneflow/core/eager/blob_object.h"

namespace oneflow {
namespace eager {

class LazyRefBlobObject final : public BlobObject {
 public:
  LazyRefBlobObject(const LazyRefBlobObject&) = delete;
  LazyRefBlobObject(LazyRefBlobObject&&) = delete;
  LazyRefBlobObject(Blob* blob)
      : BlobObject(std::make_shared<MemoryCase>(blob->mem_case()), blob->data_type()) {
    const auto& rt_blob_desc = blob->blob_desc();
    blob_desc_ = BlobDesc(rt_blob_desc.body(), rt_blob_desc.is_dynamic());
    ref_blob_ = blob;
  }
  ~LazyRefBlobObject() override = default;

  BlobDesc* mut_blob_desc() override { UNIMPLEMENTED(); }

  const Blob& blob() const override { return *ref_blob_; }
  Blob* mut_blob() override { return ref_blob_; }

  Maybe<void> TryAllocateBlobBodyMemory(DeviceCtx* device_ctx) override{
    // do nothing
    return Maybe<void>::Ok();
  };

  Maybe<void> DeallocateBlobDataPtr() override{
    // do nothing
    return Maybe<void>::Ok();
  };

  Maybe<void> TryInitBlob() override {
    // do nothing
    return Maybe<void>::Ok();
  }

 private:
  Blob* ref_blob_ = nullptr;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_REF_BLOB_OBJECT_H_
