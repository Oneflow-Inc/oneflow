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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_BATCH_MATMUL_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_BATCH_MATMUL_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/blas.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace ep {
namespace primitive {

class BatchMatmul : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchMatmul);
  BatchMatmul() = default;
  ~BatchMatmul() override = default;

  virtual void Launch(Stream* stream, size_t batch_size, size_t m, size_t n, size_t k, Scalar alpha,
                      const void* a, const void* b, Scalar beta, void* c) = 0;
};

class BatchMatmulFactory : public Factory<BatchMatmul> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchMatmulFactory);
  BatchMatmulFactory() = default;
  ~BatchMatmulFactory() override = default;

  virtual std::unique_ptr<BatchMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                           BlasTransposeType transpose_b) = 0;
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_BATCH_MATMUL_H_
