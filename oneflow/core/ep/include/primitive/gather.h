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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_GATHER_H
#define ONEFLOW_CORE_EP_PRIMITIVE_GATHER_H

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/blas.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace ep {

namespace primitive {

class Gather : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Gather);
  Gather() = default;
  ~Gather() override = default;

  virtual void Launch(Stream* stream, size_t batch_dim_size, size_t outer_dim_size,
                      size_t gather_dim_size, size_t inner_dim_size, size_t offset, const void* in,
                      size_t indices_size, const void* indices, void* out) = 0;
};

class GatherFactory : public Factory<Gather> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactory);
  GatherFactory() = default;
  ~GatherFactory() override = default;

  virtual std::unique_ptr<Gather> New(DataType params_type, DataType indices_type) = 0;
};

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_GATHER_H
