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
#ifndef ONEFLOW_USER_KERNELS_TO_CONTIGUOUS_KERNEL_H_
#define ONEFLOW_USER_KERNELS_TO_CONTIGUOUS_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

class ToContiguousUtilParam {
 protected:
  ToContiguousUtilParam(ep::Stream* stream, const ShapeView& in_shape,
                        const std::vector<int64_t>& in_stride, const char* in_dptr, char* out_dptr)
      : stream(stream),
        in_shape(in_shape),
        in_stride(in_stride),
        in_dptr(in_dptr),
        out_dptr(out_dptr) {}

  ep::Stream* stream;
  const ShapeView& in_shape;
  const std::vector<int64_t>& in_stride;
  const char* in_dptr;
  char* out_dptr;
};

class ToContiguousUtilBase : public ToContiguousUtilParam {
 public:
  ToContiguousUtilBase(ep::Stream* stream, const ShapeView& in_shape,
                       const std::vector<int64_t>& in_stride, const char* in_dptr, char* out_dptr)
      : ToContiguousUtilParam(stream, in_shape, in_stride, in_dptr, out_dptr),
        block_size(1),
        contiguous_dim(in_shape.NumAxes() - 1),
        out_stride(in_shape.NumAxes()),
        in_offset(0),
        out_offset(0),
        element_count(1) {
    for (int64_t i = contiguous_dim; i != -1; --i) {
      out_stride[i] = element_count;
      element_count *= in_shape.At(i);
    }
    for (int64_t i = contiguous_dim; i != -1; --i) {
      if (block_size == in_stride[i]) {
        block_size *= in_shape.At(i);
      } else {
        break;
      }
    }
  }

  int64_t block_size = 1;
  int64_t contiguous_dim = 0;

  DimVector out_stride;

  int64_t in_offset = 0;
  int64_t out_offset = 0;
  int64_t element_count = 1;
};

template<DeviceType, typename>
struct ToContiguousUtil : ToContiguousUtilBase {
  using ToContiguousUtilBase::ToContiguousUtilBase;

  void operator()();
};

}  // namespace oneflow

#define TO_CONTIGUOUS_COMMON_TYPES                  \
  OF_PP_MAKE_TUPLE_SEQ(bool, DataType::kBool)       \
  OF_PP_MAKE_TUPLE_SEQ(char, DataType::kChar)       \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)     \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)   \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)     \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define TO_CONTIGUOUS_CPU_TYPES                                                \
  TO_CONTIGUOUS_COMMON_TYPES OF_PP_MAKE_TUPLE_SEQ(float16, DataType::kFloat16) \
      OF_PP_MAKE_TUPLE_SEQ(bfloat16, DataType::kBFloat16)

#ifdef WITH_CUDA
#if CUDA_VERSION >= 11000
#define TO_CONTIGUOUS_CUDA_SPECIAL_TYPE          \
  OF_PP_MAKE_TUPLE_SEQ(half, DataType::kFloat16) \
  OF_PP_MAKE_TUPLE_SEQ(nv_bfloat16, DataType::kBFloat16)
#else
#define TO_CONTIGUOUS_CUDA_SPECIAL_TYPE OF_PP_MAKE_TUPLE_SEQ(half, DataType::kFloat16)
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA
#endif  // ONEFLOW_USER_KERNELS_TO_CONTIGUOUS_KERNEL_H_
