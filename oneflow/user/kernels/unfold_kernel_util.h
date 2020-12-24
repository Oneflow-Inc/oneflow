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
#ifndef ONEFLOW_USER_KERNELS_UNFOLD_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_UNFOLD_KERNEL_UTIL_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

namespace user_op {

// NDIM range: (1, 2, 3)
// SDIM range: (1, 2), 1 indicates channels_last, 2 indicates channels_first
template<typename INDEX_T, int NDIM, int SDIM>
struct UnfoldParams {
  static constexpr int kInputNDim = NDIM + 2;
  static constexpr int kOutputNDim = NDIM * 2 + 2;
  static constexpr int kInputChannelDim = (2 - SDIM) * NDIM + 1;
  static constexpr int kOutputChannelDim = (2 - SDIM) * NDIM * 2 + 1;
  static_assert(kInputChannelDim < kInputNDim, "");
  static_assert(kOutputChannelDim < kOutputNDim, "");
  UnfoldParams(const int64_t batch_size, const int64_t channels, const int64_t* spatial_dims,
               const int32_t* kernel_size, const int32_t* padding_before,
               const int32_t* padding_after, const int32_t* stride, const int32_t* dilation);
  INDEX_T elem_cnt;
  INDEX_T dims[NDIM];
  int padding[NDIM];
  int stride[NDIM];
  int dilation[NDIM];
  NdIndexOffsetHelper<INDEX_T, kInputNDim> in_index_helper;
  NdIndexOffsetHelper<INDEX_T, kOutputNDim> out_index_helper;
};

template<typename INDEX_T, int NDIM, int SDIM>
UnfoldParams<INDEX_T, NDIM, SDIM>::UnfoldParams(const int64_t batch_size, const int64_t channels,
                                                const int64_t* spatial_dims,
                                                const int32_t* kernel_size,
                                                const int32_t* padding_before,
                                                const int32_t* padding_after, const int32_t* stride,
                                                const int32_t* dilation)
    : elem_cnt(0), in_index_helper(0), out_index_helper(0) {
  INDEX_T input_dims[kInputNDim] = {0};
  INDEX_T output_dims[kOutputNDim] = {0};
  this->elem_cnt = batch_size * channels;
  input_dims[0] = batch_size;
  output_dims[0] = batch_size;
  input_dims[kInputChannelDim] = channels;
  output_dims[kOutputChannelDim] = channels;
  for (int i = 0; i < NDIM; ++i) {
    this->elem_cnt *= spatial_dims[i];
    this->dims[i] = spatial_dims[i];
    this->padding[i] = padding_before[i];
    this->stride[i] = stride[i];
    this->dilation[i] = dilation[i];
    input_dims[SDIM + i] = spatial_dims[i];
    output_dims[SDIM + i] = kernel_size[i];
    output_dims[SDIM + NDIM + i] =
        (spatial_dims[i] + padding_before[i] + padding_after[i]) / kernel_size[i];
  }
  in_index_helper = NdIndexOffsetHelper<INDEX_T, kInputNDim>(input_dims);
  out_index_helper = NdIndexOffsetHelper<INDEX_T, kOutputNDim>(output_dims);
}

// index_a format: (N, C, di, hi, wi, db, hb, wb) or (N, di, hi, wi, db, hb, wb, C)
// index_b format: (N, C, D, H, W) or (N, D, H, W, C)
// return: false indicates out-of-bound, otherwise in-bound
template<typename INDEX_T, int NDIM, int SDIM>
OF_DEVICE_FUNC bool UnfoldIndexTransform(const UnfoldParams<INDEX_T, NDIM, SDIM>& params,
                                         const INDEX_T* index_a, INDEX_T* index_b) {
  // batch dim index transform
  index_b[0] = index_a[0];
  // channel dim index transform
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  index_b[ParamType::kInputChannelDim] = index_a[ParamType::kOutputChannelDim];
  // spatial dim index transform
  bool out_bound = false;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  // D,H,W spatial dim index transform
  for (int64_t i = 0; i < NDIM; ++i) {
    INDEX_T idx = index_a[SDIM + NDIM + i] * params.stride[i]
                  + index_a[SDIM + i] * params.dilation[i] - params.padding[i];
    out_bound = out_bound && (idx < 0 || idx >= params.dims[i]);
    if (!out_bound) { index_b[i] = idx; }
  }
  return out_bound;
}

template<DeviceType device_type, typename T>
class UnfoldKernelUtil {
 public:
  static void CFirstForward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                            const Shape& out, const std::vector<int32_t>& kernel_size,
                            const std::vector<int32_t>& strides,
                            const std::vector<int32_t>& dilation_rate,
                            const std::vector<int32_t>& padding_before, const T* data_im,
                            T* data_col);

  static void CFirstBackward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                             const Shape& out, const std::vector<int32_t>& kernel_size,
                             const std::vector<int32_t>& strides,
                             const std::vector<int32_t>& dilation_rate,
                             const std::vector<int32_t>& padding_before, const T* data_col,
                             T* data_im);
};

template<DeviceType device_type, typename T, typename INDEX_T, int NDIM, int SDIM>
struct UnfoldKernelUtilV2 {
  static void Forward(DeviceCtx* ctx, const void* params, const T* input_ptr, T* output_ptr);
};

#define SPATIAL_NDIM_SEQ OF_PP_MAKE_TUPLE_SEQ(1) OF_PP_MAKE_TUPLE_SEQ(2) OF_PP_MAKE_TUPLE_SEQ(3)
#define SPATIAL_DIM_SEQ OF_PP_MAKE_TUPLE_SEQ(1) OF_PP_MAKE_TUPLE_SEQ(2)

#define INSTANTIATE_UNFOLD_KERNEL_UTIL_V2(device, dtype, itype, ndim, sdim) \
  template struct UnfoldKernelUtilV2<device, dtype, itype, ndim, sdim>;

#define INSTANTIATE_UNFOLD_KERNEL_UTIL_WITH_TYPE_PAIR(device, dtype_pair, itype_pair, ndim, sdim) \
  INSTANTIATE_UNFOLD_KERNEL_UTIL_V2(device, OF_PP_PAIR_FIRST(dtype_pair),                         \
                                    OF_PP_PAIR_FIRST(itype_pair), ndim, sdim)

#define INSTANTIATE_UNFOLD_KERNEL_UTIL_FOR_DEVICE(device)                                   \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNFOLD_KERNEL_UTIL_WITH_TYPE_PAIR, (device), \
                                   ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ,           \
                                   SPATIAL_NDIM_SEQ, SPATIAL_DIM_SEQ)

#define INSTANTIATE_UNFOLD_KERNEL_UTIL(device, dtype) \
  template class UnfoldKernelUtil<device, dtype>;

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_UNFOLD_KERNEL_UTIL_H_
