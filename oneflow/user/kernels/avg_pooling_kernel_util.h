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
#ifndef ONEFLOW_USER_KERNELS_AVG_POOLING_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_AVG_POOLING_KERNEL_UTIL_H_
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/kernel/util/numerics.cuh"
#include "oneflow/core/kernel/util/numeric_limits.cuh"
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA

namespace oneflow {

namespace {

OF_DEVICE_FUNC int64_t XPU_INT_MIN(int64_t a, int64_t b) { return a <= b ? a : b; }

OF_DEVICE_FUNC int64_t XPU_INT_MAX(int64_t a, int64_t b) { return a >= b ? a : b; }

template<typename T>
struct XPUAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) {
#if defined(__CUDA_ARCH__)
    cuda::atomic::Add(y, *x);
#else
    *y += *x;
#endif
  };
};

}  // namespace

#define AVG_POOLING_DATA_TYPE_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define AVG_POOLING_DATA_TYPE_CPU_SEQ AVG_POOLING_DATA_TYPE_SEQ

#define AVG_POOLING_DATA_TYPE_GPU_SEQ AVG_POOLING_DATA_TYPE_SEQ

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;

class AvgPoolingParams3D {
 public:
  AvgPoolingParams3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                     const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size,
                     const std::vector<int32_t>& stride, const bool ceil_mode,
                     const bool count_include_pad, const int64_t divisor_override);
  ~AvgPoolingParams3D() = default;

  const std::string& data_format() const { return data_format_; }
  const std::vector<int32_t>& padding() const { return padding_; }
  const std::vector<int32_t>& pooling_size_3d() const { return pooling_size_3d_; }
  const std::vector<int32_t>& stride_3d() const { return stride_3d_; }
  const bool& ceil_mode() const { return ceil_mode_; }
  const bool& count_include_pad() const { return count_include_pad_; }
  const int64_t& divisor_override() const { return divisor_override_; }
  const int64_t& num_batch() const { return batch_num_; }
  const int64_t& num_channel() const { return channel_num_; }

  void Reset(const ShapeView& x_shape);
  Shape GetYShape() const;
  Shape GetXShape5D() const;
  Shape GetYShape5D() const;

 private:
  int32_t dim_;
  FixedDimVector x_3d_;
  FixedDimVector y_3d_;
  std::string data_format_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> pooling_size_3d_;
  std::vector<int32_t> stride_3d_;
  bool ceil_mode_;
  bool count_include_pad_;
  int64_t divisor_override_;
  int64_t batch_num_;
  int64_t channel_num_;
};

template<DeviceType device_type, typename T>
struct AvgPoolingKernelUtil {
  static void Avgpool1dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d);

  static void Avgpool1dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d);

  static void Avgpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d);

  static void Avgpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d);

  static void Avgpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d);

  static void Avgpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d);
};

template<typename T>
OF_DEVICE_FUNC void Avgpool1dForwardCompute(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                            int64_t elem_num, const T* src, T* dest,
                                            const int32_t padding_l, const int64_t n_batch,
                                            const int64_t n_channel, const int64_t x_length,
                                            const int64_t y_length, const int32_t kernel_size_l,
                                            const int32_t stride_l, const bool count_include_pad,
                                            int64_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, l;
    index_helper.OffsetToNdIndex(num, n, c, l);

    const int64_t start_idx = (n * n_channel + c) * x_length;
    int64_t lstart = l * stride_l - padding_l;
    int64_t lend = XPU_INT_MIN(lstart + kernel_size_l, x_length + padding_l);
    const int64_t pool_size = (lend - lstart);

    lstart = XPU_INT_MAX(int64_t(0), lstart);
    lend = XPU_INT_MIN(lend, x_length);

    int64_t divide_factor;
    if (divisor_override != 0) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (lend - lstart);
      }
    }
    T sum = 0;

    for (int64_t idx = lstart; idx < lend; idx += 1) {
      const int64_t search_idx = start_idx + idx;
      sum += src[search_idx];
    }
    dest[num] = sum / divide_factor;
  }
}

template<typename T>
OF_DEVICE_FUNC void Avgpool1dBackwardCompute(
    const NdIndexOffsetHelper<int64_t, 3> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_l, const int64_t n_batch, const int64_t n_channel,
    const int64_t input_length, const int64_t output_length, const int32_t kernel_size_l,
    const int32_t stride_l, const bool count_include_pad, int64_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, l;
    index_helper.OffsetToNdIndex(num, n, c, l);

    const int64_t start_idx = (n * n_channel + c) * input_length;
    int64_t lstart = l * stride_l - padding_l;
    int64_t lend = XPU_INT_MIN(lstart + kernel_size_l, input_length + padding_l);
    const int64_t pool_size = (lend - lstart);

    lstart = XPU_INT_MAX(int64_t(0), lstart);
    lend = XPU_INT_MIN(lend, input_length);

    int64_t divide_factor;
    if (divisor_override != 0) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (lend - lstart);
      }
    }
    T grad_delta = src[num] / divide_factor;
    for (int64_t idx = lstart; idx < lend; idx += 1) {
      const int64_t search_idx = start_idx + idx;
      XPUAdd<T>::Invoke(&grad_delta, &dest[search_idx]);  // dest[search_idx] += grad_delta
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Avgpool2dForwardCompute(
    const NdIndexOffsetHelper<int64_t, 4> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_h, const int32_t padding_w, const int64_t n_batch,
    const int64_t n_channel, const int64_t x_height, const int64_t x_width, const int64_t y_height,
    const int64_t y_width, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_h, const int32_t stride_w, const bool count_include_pad,
    int64_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w;
    index_helper.OffsetToNdIndex(num, n, c, h, w);

    const int64_t start_idx = (n * n_channel + c) * x_width * x_height;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;

    int64_t hend = XPU_INT_MIN(hstart + kernel_size_h, x_height + padding_h);
    int64_t wend = XPU_INT_MIN(wstart + kernel_size_w, x_width + padding_w);
    const int64_t pool_size = (hend - hstart) * (wend - wstart);

    hstart = XPU_INT_MAX(0, hstart);
    wstart = XPU_INT_MAX(0, wstart);
    hend = XPU_INT_MIN(hend, x_height);
    wend = XPU_INT_MIN(wend, x_width);

    int64_t divide_factor;
    if (divisor_override != 0) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    T sum = 0;

    for (int64_t i = hstart; i < hend; i += 1) {
      for (int64_t j = wstart; j < wend; j += 1) {
        const int64_t tcntr = i * x_width + j;
        const int64_t search_idx = start_idx + tcntr;
        sum += src[search_idx];
      }
    }
    dest[num] = sum / divide_factor;
  }
}

template<typename T>
OF_DEVICE_FUNC void Avgpool2dBackwardCompute(
    const NdIndexOffsetHelper<int64_t, 4> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_h, const int32_t padding_w, const int64_t n_batch,
    const int64_t n_channel, const int64_t input_height, const int64_t input_width,
    const int64_t output_height, const int64_t output_width, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_h, const int32_t stride_w,
    const bool count_include_pad, int64_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w;
    index_helper.OffsetToNdIndex(num, n, c, h, w);

    const int64_t start_idx = (n * n_channel + c) * input_width * input_height;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    int64_t hend = XPU_INT_MIN(hstart + kernel_size_h, input_height + padding_h);
    int64_t wend = XPU_INT_MIN(wstart + kernel_size_w, input_width + padding_w);
    const int64_t pool_size = (hend - hstart) * (wend - wstart);

    hstart = XPU_INT_MAX(int64_t(0), hstart);
    wstart = XPU_INT_MAX(int64_t(0), wstart);
    hend = XPU_INT_MIN(hend, input_height);
    wend = XPU_INT_MIN(wend, input_width);

    int64_t divide_factor;
    if (divisor_override != 0) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    T grad_delta = src[num] / divide_factor;
    for (int64_t i = hstart; i < hend; i += 1) {
      for (int64_t j = wstart; j < wend; j += 1) {
        const int64_t tcntr = i * input_width + j;
        const int64_t search_idx = start_idx + tcntr;
        XPUAdd<T>::Invoke(&grad_delta, &dest[search_idx]);  // dest[search_idx] += grad_delta
      }
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Avgpool3dForwardCompute(
    const NdIndexOffsetHelper<int64_t, 5> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int64_t n_batch, const int64_t n_channel, const int64_t x_time, const int64_t x_height,
    const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
    const int32_t kernel_size_t, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_t, const int32_t stride_h, const int32_t stride_w,
    const bool count_include_pad, int64_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, t, h, w;
    index_helper.OffsetToNdIndex(num, n, c, t, h, w);

    const int64_t start_idx = (n * n_channel + c) * x_time * x_height * x_width;
    int64_t tstart = t * stride_t - padding_t;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    int64_t tend = XPU_INT_MIN(tstart + kernel_size_t, x_time + padding_t);
    int64_t hend = XPU_INT_MIN(hstart + kernel_size_h, x_height + padding_h);
    int64_t wend = XPU_INT_MIN(wstart + kernel_size_w, x_width + padding_w);
    const int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

    tstart = XPU_INT_MAX(int64_t(0), tstart);
    hstart = XPU_INT_MAX(int64_t(0), hstart);
    wstart = XPU_INT_MAX(int64_t(0), wstart);
    tend = XPU_INT_MIN(tend, x_time);
    hend = XPU_INT_MIN(hend, x_height);
    wend = XPU_INT_MIN(wend, x_width);

    int64_t divide_factor;
    if (divisor_override != 0) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
      }
    }
    T sum = 0;

    for (int64_t i = tstart; i < tend; i += 1) {
      for (int64_t j = hstart; j < hend; j += 1) {
        for (int64_t k = wstart; k < wend; k += 1) {
          const int64_t tcntr = i * x_height * x_width + j * x_width + k;
          const int64_t search_idx = start_idx + tcntr;
          sum += src[search_idx];
        }
      }
    }
    dest[num] = sum / divide_factor;
  }
}

template<typename T>
OF_DEVICE_FUNC void Avgpool3dBackwardCompute(
    const NdIndexOffsetHelper<int64_t, 5> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int64_t n_batch, const int64_t n_channel, const int64_t x_time, const int64_t x_height,
    const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
    const int32_t kernel_size_t, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_t, const int32_t stride_h, const int32_t stride_w,
    const bool count_include_pad, int64_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, t, h, w;
    index_helper.OffsetToNdIndex(num, n, c, t, h, w);

    const int64_t start_idx = (n * n_channel + c) * x_time * x_width * x_height;
    int64_t tstart = t * stride_t - padding_t;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    int64_t tend = XPU_INT_MIN(tstart + kernel_size_t, x_time + padding_t);
    int64_t hend = XPU_INT_MIN(hstart + kernel_size_h, x_height + padding_h);
    int64_t wend = XPU_INT_MIN(wstart + kernel_size_w, x_width + padding_w);
    const int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

    tstart = XPU_INT_MAX(int64_t(0), tstart);
    hstart = XPU_INT_MAX(int64_t(0), hstart);
    wstart = XPU_INT_MAX(int64_t(0), wstart);
    tend = XPU_INT_MIN(tend, x_time);
    hend = XPU_INT_MIN(hend, x_height);
    wend = XPU_INT_MIN(wend, x_width);

    int64_t divide_factor;
    if (divisor_override != 0) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
      }
    }
    T grad_delta = src[num] / divide_factor;

    for (int64_t i = tstart; i < tend; i += 1) {
      for (int64_t j = hstart; j < hend; j += 1) {
        for (int64_t k = wstart; k < wend; k += 1) {
          const int64_t tcntr = i * x_height * x_width + j * x_width + k;
          const int64_t search_idx = start_idx + tcntr;
          XPUAdd<T>::Invoke(&grad_delta, &dest[search_idx]);  // dest[search_idx] += grad_delta
        }
      }
    }
  }
}

#define INSTANTIATE_AVG_POOLING_KERNEL_UTIL(device_type_v, dtype_pair) \
  template struct AvgPoolingKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_AVG_POOLING_KERNEL_UTIL_H_