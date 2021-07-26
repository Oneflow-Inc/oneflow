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
#include "oneflow/user/kernels/constantpad_kernel_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace user_op {

template<typename IN_T>
struct ConstantPad1dFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& x_shape,
                  const ShapeView& y_shape, const std::vector<int64_t>& padding,
                  IN_T constant_value) {
    // for NCW format input tensor, index of n, c, w is 0, 1, 2
    const int64_t c_idx = 1;
    const int64_t w_idx = 2;
    // padding vector: [left, right]
    DoConstantPad1d<IN_T>(src, dest, index_helper, y_shape.Count(0), y_shape.At(c_idx),
                          y_shape.At(w_idx), x_shape.At(w_idx), padding[0], constant_value);
  }
};

template<typename IN_T>
struct ConstantPad1dGradFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& dy_shape,
                  const ShapeView& dx_shape, const std::vector<int64_t>& padding) {
    const int64_t c_idx = 1;
    const int64_t w_idx = 2;
    DoConstantPad1dGrad<IN_T>(src, dest, index_helper, dy_shape.Count(0), dy_shape.At(c_idx),
                              dy_shape.At(w_idx), dx_shape.At(w_idx), padding[0]);
  }
};

template<typename IN_T>
struct ConstantPad3dFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& x_shape,
                  const ShapeView& y_shape, const std::vector<int64_t>& padding,
                  IN_T constant_value) {
    // for NCDHW format input tensor, index of n, c, d, h, w is 0, 1, 2, 3, 4
    const int64_t c_idx = 1;
    const int64_t d_idx = 2;
    const int64_t h_idx = 3;
    const int64_t w_idx = 4;
    // padding vector: [left, right, top, bottom, front, back]
    DoConstantPad3d<IN_T>(src, dest, index_helper, y_shape.Count(0), y_shape.At(c_idx),
                          y_shape.At(d_idx), y_shape.At(h_idx), y_shape.At(w_idx),
                          x_shape.At(d_idx), x_shape.At(h_idx), x_shape.At(w_idx), padding[4],
                          padding[0], padding[2], constant_value);
  }
};

template<typename IN_T>
struct ConstantPad3dGradFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& dy_shape,
                  const ShapeView& dx_shape, const std::vector<int64_t>& padding) {
    const int64_t c_idx = 1;
    const int64_t d_idx = 2;
    const int64_t h_idx = 3;
    const int64_t w_idx = 4;
    DoConstantPad3dGrad<IN_T>(src, dest, index_helper, dy_shape.Count(0), dy_shape.At(c_idx),
                              dy_shape.At(d_idx), dy_shape.At(h_idx), dy_shape.At(w_idx),
                              dx_shape.At(d_idx), dx_shape.At(h_idx), dx_shape.At(w_idx),
                              padding[4], padding[0], padding[2]);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CONSTANT_PAD_FUNCTOR, (DeviceType::kCPU),
                                 PADDING_DATA_TYPE_CPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CONSTANT_PAD_GRAD_FUNCTOR, (DeviceType::kCPU),
                                 PADDING_DATA_TYPE_CPU_SEQ);

}  // namespace user_op
}  // namespace oneflow
