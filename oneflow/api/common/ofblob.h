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

#ifndef ONEFLOW_API_COMMON_OFBLOB_H_
#define ONEFLOW_API_COMMON_OFBLOB_H_

#include "oneflow/core/common/just.h"
#include "oneflow/core/register/ofblob.h"

namespace oneflow {

template<typename T>
struct BlobBufferCopyUtil {
  static Maybe<void> From(uint64_t of_blob_ptr, const T* buf_ptr, size_t size) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->AutoMemCopyFrom<T>(buf_ptr, size);
    return Maybe<void>::Ok();
  }

  static Maybe<void> To(uint64_t of_blob_ptr, T* buf_ptr, size_t size) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->AutoMemCopyTo<T>(buf_ptr, size);
    return Maybe<void>::Ok();
  }
};

template<>
struct BlobBufferCopyUtil<void> {
  static Maybe<void> From(uint64_t of_blob_ptr, const void* buf_ptr, size_t size) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->AutoMemCopyFrom<void>(buf_ptr, size);
    return Maybe<void>::Ok();
  }

  static Maybe<void> To(uint64_t of_blob_ptr, void* buf_ptr, size_t size) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->AutoMemCopyTo<void>(buf_ptr, size);
    return Maybe<void>::Ok();
  }
};

}  // namespace oneflow

#endif  // !ONEFLOW_API_COMMON_OFBLOB_H_
