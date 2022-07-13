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
#ifndef ONEFLOW_CORE_EP_INCLUDE_PRIMITIVE_GATHER_H_
#define ONEFLOW_CORE_EP_INCLUDE_PRIMITIVE_GATHER_H_

#include "oneflow/core/ep/include/primitive/primitive.h"

namespace oneflow {
namespace ep {
namespace primitive {
class Gather : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Gather);
  Gather() = default;
  ~Gather() override = default;
  virtual void Launch(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                      int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
                      int64_t num_indices, const void* indice, void* output) = 0;
  virtual void Launch(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                      int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
                      int64_t num_indices, const void* indice, int64_t offset, void* output) = 0;
};
class GatherFactory : public Factory<Gather> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactory);
  GatherFactory() = default;
  ~GatherFactory() override = default;
  virtual std::unique_ptr<Gather> New(DataType data_dtype, DataType indice_type) = 0;
};

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EP_INCLUDE_PRIMITIVE_GATHER_H_
