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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_PERMUTE_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_PERMUTE_H_

#include "oneflow/core/ep/include/primitive/primitive.h"

namespace oneflow {

namespace ep {
namespace primitive {

class Permute : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Permute);
  Permute() = default;
  ~Permute() override = default;

  virtual void Launch(Stream* stream, DataType data_type, size_t num_dims, const int64_t* src_dims,
                      const void* src, const int* permutation, void* dst) = 0;
};

class PermuteFactory : public Factory<Permute> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteFactory);
  PermuteFactory() = default;
  ~PermuteFactory() override = default;

  virtual std::unique_ptr<Permute> New(size_t max_num_dims) = 0;
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_PERMUTE_H_
