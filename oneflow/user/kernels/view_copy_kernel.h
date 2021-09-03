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

#ifndef ONEFLOW_USER_KERNELS_VIEW_COPY_KERNEL_H_
#define ONEFLOW_USER_KERNELS_VIEW_COPY_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

class ViewCopyUtilParam {
 protected:
  ViewCopyUtilParam(const DeviceCtx* ctx, const ShapeView& in_shape,
                    const std::vector<int64_t>& in_stride, const char* in_dptr, char* out_dptr)
      : ctx(ctx), in_shape(in_shape), in_stride(in_stride), in_dptr(in_dptr), out_dptr(out_dptr) {}

  const DeviceCtx* ctx;
  const ShapeView& in_shape;
  const std::vector<int64_t>& in_stride;
  const char* in_dptr;
  char* out_dptr;
};

class ViewCopyUtilAttach;

class ViewCopyUtilBase : public ViewCopyUtilParam {
 public:
  ViewCopyUtilBase(const DeviceCtx* ctx, const ShapeView& in_shape,
                   const std::vector<int64_t>& in_stride, const char* in_dptr, char* out_dptr)
      : ViewCopyUtilParam(ctx, in_shape, in_stride, in_dptr, out_dptr),
        contiguous_block_size(1),
        contiguous_dim(in_shape.NumAxes() - 1),
        out_stride(in_shape.NumAxes()),
        index(contiguous_dim + 1),
        in_offset(0),
        out_offset(0) {
    for (; contiguous_dim != -1; --contiguous_dim) {
      if (contiguous_block_size == in_stride[contiguous_dim]) {
        contiguous_block_size *= in_shape.At(contiguous_dim);
      } else {
        break;
      }
    }
  }

 protected:
  int64_t init_out_stride() {
    int64_t sum = 1;
    for (int64_t i = out_stride.size() - 1; i != -1; --i) {
      out_stride[i] = sum;
      sum *= in_shape.At(i);
    }
    return sum;
  }

  void init_index() { std::fill(index.begin(), index.end(), 0); }

  bool next_index() {
    int64_t i = contiguous_dim;
    for (; i != -1; --i) {
      if (index[i] == in_shape.At(i) - 1) {
        in_offset -= in_stride[i] * index[i];
        out_offset -= out_stride[i] * index[i];
        index[i] = 0;
      } else {
        index[i]++;
        in_offset += in_stride[i];
        out_offset += out_stride[i];
        break;
      }
    }
    return i == -1;
  }

  int64_t contiguous_block_size;
  int64_t contiguous_dim;

  StrideVector out_stride;
  DimVector index;

  int64_t in_offset;
  int64_t out_offset;
};

template<DeviceType, typename>
struct ViewCopyUtil : ViewCopyUtilBase {
  using ViewCopyUtilBase::ViewCopyUtilBase;

  void operator()();
};

}  // namespace oneflow

#define VIEW_COPY_TYPES         \
  OF_PP_MAKE_TUPLE_SEQ(float)   \
  OF_PP_MAKE_TUPLE_SEQ(double)  \
  OF_PP_MAKE_TUPLE_SEQ(int32_t) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t) \
  OF_PP_MAKE_TUPLE_SEQ(int8_t)  \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t)
#define VIEW_COPY_GPU_SPECIAL_TYPE OF_PP_MAKE_TUPLE_SEQ(float16)

#endif  // ONEFLOW_USER_KERNELS_VIEW_COPY_KERNEL_H_
