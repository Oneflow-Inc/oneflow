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
#ifndef ONEFLOW_CORE_PRIMITIVE_MEMORY_COPY_ND_H_
#define ONEFLOW_CORE_PRIMITIVE_MEMORY_COPY_ND_H_

#include "oneflow/core/primitive/include/primitive.h"

namespace oneflow {

namespace primitive {

class MemoryCopyNd : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopyNd);
  MemoryCopyNd() = default;
  ~MemoryCopyNd() override = default;

  virtual void Launch(StreamContext* stream_ctx, DataType data_type, size_t num_dims, void* dst,
                      const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                      const int64_t* src_dims, const int64_t* src_pos,
                      const int64_t* extent) const = 0;
};

class MemoryCopyNdFactory : public Factory<MemoryCopyNd> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopyNdFactory);
  MemoryCopyNdFactory() = default;
  ~MemoryCopyNdFactory() override = default;

  virtual std::unique_ptr<MemoryCopyNd> New() = 0;
};

}  // namespace primitive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_MEMORY_COPY_ND_H_
