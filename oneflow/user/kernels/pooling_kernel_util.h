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
#ifndef ONEFLOW_USER_KERNELS_POOLING_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_POOLING_KERNEL_UTIL_H_
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

#define POOLING_DATA_TYPE_CPU_SEQ                 \
  FLOATING_DATA_TYPE_SEQ                          \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define POOLING_DATA_TYPE_GPU_SEQ POOLING_DATA_TYPE_CPU_SEQ

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;
typedef fixed_vector<int32_t, SHAPE_MAX_AXIS_SIZE> FixedVector;

class PoolingParams3D {
 public:
  PoolingParams3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                  const std::string& padding, const std::vector<int32_t>& padding_before,
                  const std::vector<int32_t>& padding_after,
                  const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride,
                  const std::vector<int32_t>& dilation, const bool return_indices,
                  const bool ceil_mode);
  ~PoolingParams3D() = default;
  void Reset(const ShapeView& x_shape);

  Shape GetYShape() const;
  Shape GetXShape5D() const;
  Shape GetYShape5D() const;

  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& strides_3d() const { return strides_3d_; }
  const std::vector<int32_t>& padding_before_3d() const { return padding_before_3d_; }
  const std::vector<int32_t>& padding_after_3d() const { return padding_after_3d_; }
  const std::vector<int32_t>& dilation_3d() const { return dilation_3d_; }

 private:
  int32_t dim_;
  FixedDimVector x_3d_;
  FixedDimVector y_3d_;
  std::vector<int32_t> pool_size_3d_;
  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> padding_before_3d_;
  std::vector<int32_t> padding_after_3d_;
  std::vector<int32_t> dilation_3d_;
  std::string data_format_;
  std::string padding_;
  bool return_indices_;
  bool ceil_mode_;
  int64_t batch_num_;
  int64_t channel_num_;
};

struct PoolKernelState final : public user_op::OpKernelState {
  PoolingParams3D params_3d;
  bool is_dynamic;
  PoolKernelState(PoolingParams3D params_3d, bool is_dynamic)
      : params_3d(params_3d), is_dynamic(is_dynamic) {}
  const PoolingParams3D& GetParams3D() { return params_3d; }
  void Update(const ShapeView& x_shape) {
    if (is_dynamic) { params_3d.Reset(x_shape); }
  }
};

template<DeviceType device_type, typename T>
struct PoolingKernelUtil {
  static void Maxpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const std::vector<int32_t> padding_before, const int64_t n_batch,
                               const int64_t n_channel, const int64_t x_height,
                               const int64_t x_width, const int64_t y_height, const int64_t y_width,
                               const std::vector<int32_t> kernel_size,
                               const std::vector<int32_t> stride,
                               const std::vector<int32_t> dilation, const bool return_indices,
                               const bool ceil_mode);

  static void Maxpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const int64_t n_batch,
                                const int64_t n_channel, const int64_t src_height,
                                const int64_t src_width, const int64_t dst_height,
                                const int64_t dst_width, const bool return_indices,
                                const bool ceil_mode);

  static void Maxpool3dForward(
      DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper, const int64_t elem_num,
      const T* src, T* dest, int64_t* indice_ptr, const std::vector<int32_t> padding_before,
      const int64_t n_batch, const int64_t n_channel, const int64_t x_time, const int64_t x_height,
      const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
      const std::vector<int32_t> kernel_size, const std::vector<int32_t> stride,
      const std::vector<int32_t> dilation, const bool return_indices, const bool ceil_mode);

  static void Maxpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const int64_t n_batch,
                                const int64_t n_channel, const int64_t src_time,
                                const int64_t src_height, const int64_t src_width,
                                const int64_t dst_time, const int64_t dst_height,
                                const int64_t dst_width, const bool return_indices,
                                const bool ceil_mode);
};

template<typename T>
OF_DEVICE_FUNC void Maxpool2dFarwardCompute(
    const NdIndexOffsetHelper<int64_t, 4> index_helper, int64_t elem_num, T maxval, const T* src,
    T* dest, int64_t* indice_ptr, const int32_t padding_h, const int32_t padding_w,
    const int64_t n_batch, const int64_t n_channel, const int64_t x_height, const int64_t x_width,
    const int64_t y_height, const int64_t y_width, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_h, const int32_t dilation_w, const bool return_indices,
    const bool ceil_mode) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w, coords[4] = {0};
    index_helper.OffsetToNdIndex(num, coords);
    n = coords[0];
    c = coords[1];
    h = coords[2];
    w = coords[3];

    int64_t xstart = n * n_channel * x_width * x_height;
    int64_t ystart = n * n_channel * y_height * y_width;
    int64_t ip = xstart + c * x_width * x_height;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    int64_t hend, wend;
    if ((hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height) {
      hend = (hstart + (kernel_size_h - 1) * dilation_h + 1);
    } else {
      hend = x_height;
    }
    if ((wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width) {
      wend = (wstart + (kernel_size_w - 1) * dilation_w + 1);
    } else {
      wend = x_width;
    }
    while (hstart < 0) hstart += dilation_h;
    while (wstart < 0) wstart += dilation_w;

    /* local pointers */
    int64_t dest_idx = ystart + c * y_width * y_height + h * y_width + w;
    int64_t indp = ystart + c * y_width * y_height + h * y_width + w;
    /* compute local max: */
    int64_t maxindex = hstart * x_width + wstart;
    int64_t src_idx;
    /* T maxval = -std::numeric_limits<T>::infinity(); */
    for (int64_t i = hstart; i < hend; i += dilation_h) {
      for (int64_t j = wstart; j < wend; j += dilation_w) {
        int64_t tcntr = i * x_width + j;
        int64_t search_idx = ip + tcntr;
        T val = src[search_idx];
        if ((val > maxval) || std::isnan(val))
        // if (val > maxval)
        {
          maxval = val;
          maxindex = tcntr;
          src_idx = search_idx;
        }
      }
    }
    /* set output to local max */
    dest[dest_idx] = src[src_idx];
    /* store location of max */
    indice_ptr[indp] = maxindex;
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool2dBackwardCompute(const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                             const int64_t elem_num, const T* src, T* dest,
                                             const int64_t* indice_ptr, const int64_t n_batch,
                                             const int64_t n_channel, const int64_t src_height,
                                             const int64_t src_width, const int64_t dst_height,
                                             const int64_t dst_width, const bool return_indices,
                                             const bool ceil_mode) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w;
    int64_t coord_dx[4] = {0};
    index_helper.OffsetToNdIndex(num, coord_dx);
    n = coord_dx[0];
    c = coord_dx[1];
    h = coord_dx[2];
    w = coord_dx[3];

    int64_t src_start = n * n_channel * src_height * src_width;
    int64_t dst_start = n * n_channel * dst_height * dst_width;

    int64_t ip = dst_start + c * dst_width * dst_height;
    int64_t src_idx = src_start + c * src_height * src_width + h * src_width + w;
    int64_t indice_idx = c * src_height * src_width + h * src_width + w;
    int64_t dest_idx = ip + indice_ptr[indice_idx];
    if (dest_idx != -1) {
      /* update gradient */
      dest[dest_idx] += src[src_idx];
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool3dFarwardCompute(
    const NdIndexOffsetHelper<int64_t, 5> index_helper, int64_t elem_num, T maxval, const T* src,
    T* dest, int64_t* indice_ptr, const int32_t padding_t, const int32_t padding_h,
    const int32_t padding_w, const int64_t n_batch, const int64_t n_channel, const int64_t x_time,
    const int64_t x_height, const int64_t x_width, const int64_t y_time, const int64_t y_height,
    const int64_t y_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const int32_t dilation_t, const int32_t dilation_h,
    const int32_t dilation_w, const bool return_indices, const bool ceil_mode) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, t, c, h, w, coords[5] = {0};
    index_helper.OffsetToNdIndex(num, coords);
    n = coords[0];
    c = coords[1];
    t = coords[2];
    h = coords[3];
    w = coords[4];

    int64_t xstart = n * n_channel * x_time * x_width * x_height;
    int64_t ystart = n * n_channel * y_time * y_height * y_width;
    int64_t ip = xstart + c * x_time * x_width * x_height;
    int64_t tstart = t * stride_t - padding_t;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    int64_t tend, hend, wend;
    if ((tstart + (kernel_size_t - 1) * dilation_t + 1) <= x_time) {
      tend = (tstart + (kernel_size_t - 1) * dilation_t + 1);
    } else {
      tend = x_time;
    }
    if ((hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height) {
      hend = (hstart + (kernel_size_h - 1) * dilation_h + 1);
    } else {
      hend = x_height;
    }
    if ((wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width) {
      wend = (wstart + (kernel_size_w - 1) * dilation_w + 1);
    } else {
      wend = x_width;
    }
    while (tstart < 0) tstart += dilation_t;
    while (hstart < 0) hstart += dilation_h;
    while (wstart < 0) wstart += dilation_w;

    /* local pointers */
    int64_t dest_idx =
        ystart + c * y_time * y_width * y_height + t * y_width * y_height + h * y_width + w;
    int64_t indp =
        ystart + c * y_time * y_width * y_height + t * y_width * y_height + h * y_width + w;
    /* compute local max: */
    int64_t maxindex = tstart * x_height * x_width + hstart * x_width + wstart;
    int64_t src_idx;
    for (int64_t z = tstart; z < tend; z += dilation_t) {
      for (int64_t i = hstart; i < hend; i += dilation_h) {
        for (int64_t j = wstart; j < wend; j += dilation_w) {
          int64_t tcntr = z * x_height * x_width + i * x_width + j;
          int64_t search_idx = ip + tcntr;
          T val = src[search_idx];
          if ((val > maxval) || std::isnan(val)) {
            maxval = val;
            maxindex = tcntr;
            src_idx = search_idx;
          }
        }
      }
      /* set output to local max */
      dest[dest_idx] = src[src_idx];
      /* store location of max */
      indice_ptr[indp] = maxindex;
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool3dBackwardCompute(const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                             const int64_t elem_num, const T* src, T* dest,
                                             const int64_t* indice_ptr, const int64_t n_batch,
                                             const int64_t n_channel, const int64_t src_time,
                                             const int64_t src_height, const int64_t src_width,
                                             const int64_t dst_time, const int64_t dst_height,
                                             const int64_t dst_width, const bool return_indices,
                                             const bool ceil_mode) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, t, h, w;
    int64_t coord_dx[5] = {0};
    index_helper.OffsetToNdIndex(num, coord_dx);
    n = coord_dx[0];
    c = coord_dx[1];
    t = coord_dx[2];
    h = coord_dx[3];
    w = coord_dx[4];

    int64_t src_start = n * n_channel * src_time * src_height * src_width;
    int64_t dst_start = n * n_channel * dst_time * dst_height * dst_width;

    int64_t ip = dst_start + c * dst_time * dst_width * dst_height;
    int64_t src_idx = src_start + c * src_time * src_height * src_width + h * src_width + w;
    int64_t indice_idx = c * src_height * src_width + h * src_width + w;
    int64_t dest_idx = ip + indice_ptr[indice_idx];
    if (dest_idx != -1) {
      /* update gradient */
      dest[dest_idx] += src[src_idx];
    }
  }
}

#define INSTANTIATE_POOLING_KERNEL_UTIL(device_type_v, dtype_pair) \
  template struct PoolingKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_POOLING_KERNEL_UTIL_H_
