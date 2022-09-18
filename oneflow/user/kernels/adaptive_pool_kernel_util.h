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
#ifndef _ONEFLOW_USER_KERNELS_ADAPTIVE_POOL_UTIL_H_
#define _ONEFLOW_USER_KERNELS_ADAPTIVE_POOL_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/utils/pool_util.h"

namespace oneflow {

namespace {

inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::floor((float)(a * c) / b);
}

inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::ceil((float)((a + 1) * c) / b);
}

#define START_IND(a, b, c) (int)std::floor((float)(a * c) / b)
#define END_IND(a, b, c) (int)std::ceil((float)((a + 1) * c) / b)

#define START_IND_INT(a, b, c) ((a * c) / b)
#define END_IND_INT(a, b, c) (((a + 1) * c + b - 1) / b)

inline Shape GetShape5D(const Shape& shape, const std::string& data_format, int32_t dim) {
  FixedDimVector shape_3d = {GetInDim(shape, data_format, 0, dim),
                             GetInDim(shape, data_format, 1, dim),
                             GetInDim(shape, data_format, 2, dim)};
  return Shape({shape.At(0), shape.At(1), shape_3d.at(0), shape_3d.at(1), shape_3d.at(2)});
}

}  // namespace
}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ADAPTIVE_POOL_UTIL_H_
