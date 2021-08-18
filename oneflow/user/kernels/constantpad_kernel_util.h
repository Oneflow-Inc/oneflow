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
#ifndef ONEFLOW_USER_KERNELS_PAD_KERNELS_UTIL_H_
#define ONEFLOW_USER_KERNELS_PAD_KERNELS_UTIL_H_
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

#define PADDING_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define PADDING_DATA_TYPE_GPU_SEQ \
  FLOAT16_DATA_TYPE_SEQ           \
  PADDING_DATA_TYPE_CPU_SEQ

namespace user_op {

template<typename T>
struct DeviceAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) {
#if defined(__CUDA_ARCH__)
    cuda::atomic::Add(y, *x);
#else
    *y += *x;
#endif
  };
};

template<DeviceType device_type, typename IN_T>
struct ConstantPad1dFunctor final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& x_shape,
                  const ShapeView& y_shape, const std::vector<int64_t>& padding,
                  IN_T constant_value);
};

template<DeviceType device_type, typename IN_T>
struct ConstantPad1dGradFunctor final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& dy_shape,
                  const ShapeView& dx_shape, const std::vector<int64_t>& padding);
};

template<DeviceType device_type, typename IN_T>
struct ConstantPad3dFunctor final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& x_shape,
                  const ShapeView& y_shape, const std::vector<int64_t>& padding,
                  IN_T constant_value);
};

template<DeviceType device_type, typename IN_T>
struct ConstantPad3dGradFunctor final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& dy_shape,
                  const ShapeView& dx_shape, const std::vector<int64_t>& padding);
};

template<typename IN_T>
OF_DEVICE_FUNC void DoConstantPad1d(const IN_T* src, IN_T* dest,
                                    const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                    int64_t elem_num, int64_t n_channel, int64_t y_width,
                                    int64_t x_width, int64_t pad_left, IN_T constant_value) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, w;
    index_helper.OffsetToNdIndex(num, n, c, w);

    const int64_t src_num = n_channel * x_width;
    if (w >= pad_left && w < x_width + pad_left) {
      const int64_t len_w = w - pad_left;

      const int64_t src_index = n * src_num + c * x_width + len_w;
      dest[num] = src[src_index];
    } else {
      dest[num] = constant_value;
    }
  }
}

template<typename IN_T>
OF_DEVICE_FUNC void DoConstantPad1dGrad(const IN_T* src, IN_T* dest,
                                        const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                        int64_t elem_num, int64_t n_channel, int64_t dy_width,
                                        int64_t dx_width, int64_t pad_left) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, w;
    index_helper.OffsetToNdIndex(num, n, c, w);

    const int64_t dest_num = n_channel * dx_width;
    if (w >= pad_left && w < dx_width + pad_left) {
      const int64_t len_w = w - pad_left;
      const int64_t dest_index = n * dest_num + c * dx_width + len_w;
      DeviceAdd<IN_T>::Invoke(src + num, dest + dest_index);
    }
  }
}

template<typename IN_T>
OF_DEVICE_FUNC void DoConstantPad3d(const IN_T* src, IN_T* dest,
                                    const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                    int64_t elem_num, int64_t n_channel, int64_t y_depth,
                                    int64_t y_height, int64_t y_width, int64_t x_depth,
                                    int64_t x_height, int64_t x_width, int64_t pad_front,
                                    int64_t pad_left, int64_t pad_top, IN_T constant_value) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, d, h, w;
    index_helper.OffsetToNdIndex(num, n, c, d, h, w);

    const int64_t src_num = n_channel * x_depth * x_height * x_width;
    if (pad_front <= d && d < pad_front + x_depth && w >= pad_left && w < x_width + pad_left
        && h >= pad_top && h < x_height + pad_top) {
      const int64_t len_w = w - pad_left;
      const int64_t len_h = h - pad_top;
      const int64_t len_d = d - pad_front;
      const int64_t src_index = n * src_num + c * x_depth * x_width * x_height
                                + len_d * x_height * x_width + len_h * x_width + len_w;
      dest[num] = src[src_index];
    } else {
      dest[num] = constant_value;
    }
  }
}

template<typename IN_T>
OF_DEVICE_FUNC void DoConstantPad3dGrad(const IN_T* src, IN_T* dest,
                                        const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                        int64_t elem_num, int64_t n_channel, int64_t dy_depth,
                                        int64_t dy_height, int64_t dy_width, int64_t dx_depth,
                                        int64_t dx_height, int64_t dx_width, int64_t pad_front,
                                        int64_t pad_left, int64_t pad_top) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, d, h, w;
    index_helper.OffsetToNdIndex(num, n, c, d, h, w);

    const int64_t dest_num = n_channel * dx_depth * dx_height * dx_width;
    if (pad_front <= d && d < pad_front + dx_depth && w >= pad_left && w < dx_width + pad_left
        && h >= pad_top && h < dx_height + pad_top) {
      const int64_t len_d = d - pad_front;
      const int64_t len_w = w - pad_left;
      const int64_t len_h = h - pad_top;
      const int64_t dest_index = n * dest_num + c * dx_depth * dx_width * dx_height
                                 + len_d * dx_width * dx_height + len_h * dx_width + len_w;

      DeviceAdd<IN_T>::Invoke(src + num, dest + dest_index);
    }
  }
}

#define INSTANTIATE_CONSTANT_PAD_FUNCTOR(device_type_v, dtype_pair)                  \
  template struct ConstantPad1dFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>; \
  template struct ConstantPad3dFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

#define INSTANTIATE_CONSTANT_PAD_GRAD_FUNCTOR(device_type_v, dtype_pair)                 \
  template struct ConstantPad1dGradFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>; \
  template struct ConstantPad3dGradFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_PAD_KERNELS_UTIL_H_
