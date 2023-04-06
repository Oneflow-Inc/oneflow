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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_WHERE_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_WHERE_H_

#include "oneflow/core/ep/include/primitive/primitive.h"

namespace oneflow {
namespace ep {
namespace primitive {

class Where : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Where);
  Where() = default;
  ~Where() override = default;

  virtual void Launch(Stream* stream, size_t num_cond_dims, const int64_t* cond_dims,
                      const void* cond, size_t num_x_dims, const int64_t* x_dims, const void* x,
                      size_t num_y_dims, const int64_t* y_dims, const void* y, void* z) = 0;
};

class WhereFactory : public Factory<Where> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereFactory);
  WhereFactory() = default;
  ~WhereFactory() override = default;

  virtual std::unique_ptr<Where> New(DataType cond_type, DataType data_type,
                                     size_t max_num_dims) = 0;
};

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_WHERE_H_
