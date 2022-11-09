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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_COPY_ND_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_COPY_ND_H_

#include "oneflow/core/ep/include/primitive/primitive.h"

namespace oneflow {

namespace ep {
namespace primitive {

class CopyNd : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNd);
  CopyNd() = default;
  ~CopyNd() override = default;

  virtual void Launch(Stream* stream, DataType data_type, size_t num_dims, void* dst,
                      const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                      const int64_t* src_dims, const int64_t* src_pos,
                      const int64_t* extent) const = 0;
};

class CopyNdFactory : public Factory<CopyNd> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdFactory);
  CopyNdFactory() = default;
  ~CopyNdFactory() override = default;

  virtual std::unique_ptr<CopyNd> New(size_t max_num_dims) = 0;
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_COPY_ND_H_
