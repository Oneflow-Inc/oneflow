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
#ifndef ONEFLOW_CAMBRICON_KERNELS_SLICE_UTIL_H_
#define ONEFLOW_CAMBRICON_KERNELS_SLICE_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {
namespace mlu {

inline int64_t RegulateSliceStart(int64_t start, int64_t size) {
  // slice start must be in range [-size, size)
  // after changing to positive order it should be in range [0, size)
  start = std::min(std::max(start, -size), size - 1);
  return (start < 0) ? (start + size) : start;
}

inline int64_t RegulateSliceStop(int64_t stop, int64_t size) {
  // slice stop must be in range [-size-1, size]
  // after changing to positive order it should be in range [-1, size]
  stop = std::min(std::max(stop, -size - 1), size);
  return (stop < 0) ? (stop + size) : stop;
}

constexpr size_t kSliceMaxDims = 8;

struct SliceParams {
  int64_t ndim = 0;
  int64_t dims[kSliceMaxDims]{0};
  int64_t stride[kSliceMaxDims]{0};
  int64_t start[kSliceMaxDims]{0};
  int64_t step[kSliceMaxDims]{0};
  int64_t size[kSliceMaxDims]{0};

  int64_t elem_cnt() const {
    if (ndim == 0) { return 0; }
    int64_t elem_cnt = 1;
    FOR_RANGE(int, i, 0, ndim) { elem_cnt *= size[i]; }
    return elem_cnt;
  }

  bool IsFullSlice(int dim) const {
    CHECK_GE(dim, 0);
    CHECK_LT(dim, ndim);
    if (step[dim] != 1) { return false; }
    if (start[dim] != 0) { return false; }
    if (size[dim] != dims[dim]) { return false; }
    return true;
  }
};

class SliceKernelUtil {
 public:
  static void Forward(ep::Stream* stream, const SliceParams& params, DataType data_type,
                      const void* entire, void* sliced);
  static void Forward(ep::Stream* stream, const SliceParams& entire_params,
                      const SliceParams& sliced_params, DataType data_type, const void* entire,
                      void* sliced);
};

}  // namespace mlu
}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_KERNELS_SLICE_UTIL_H_
