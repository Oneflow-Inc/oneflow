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
#ifndef ONEFLOW_USER_KERNELS_AVG_POOL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_AVG_POOL_KERNEL_UTIL_H_
#include "oneflow/core/ep/include/stream.h"
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

template<typename T>
OF_DEVICE_FUNC T XPU_INT_MIN(T a, T b) {
  return a <= b ? a : b;
}

template<typename T>
OF_DEVICE_FUNC T XPU_INT_MAX(T a, T b) {
  return a >= b ? a : b;
}

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

#define AVG_POOL_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define AVG_POOL_DATA_TYPE_CPU_SEQ AVG_POOL_DATA_TYPE_SEQ

#define AVG_POOL_DATA_TYPE_CUDA_SEQ AVG_POOL_DATA_TYPE_SEQ

#define AVG_POOL_IDX_DATA_TYPE_SEQ                \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

typedef small_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;

class AvgPoolParams3D {
 public:
  AvgPoolParams3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                  const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size,
                  const std::vector<int32_t>& stride, const bool ceil_mode,
                  const bool count_include_pad, const int32_t divisor_override);
  ~AvgPoolParams3D() = default;

  const std::string& data_format() const { return data_format_; }
  const std::vector<int32_t>& padding() const { return padding_; }
  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& stride_3d() const { return stride_3d_; }
  const bool& ceil_mode() const { return ceil_mode_; }
  const bool& count_include_pad() const { return count_include_pad_; }
  const int32_t& divisor_override() const { return divisor_override_; }
  const int32_t& num_batch() const { return batch_num_; }
  const int32_t& num_channel() const { return channel_num_; }

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
  std::vector<int32_t> pool_size_3d_;
  std::vector<int32_t> stride_3d_;
  bool ceil_mode_;
  bool count_include_pad_;
  int32_t divisor_override_;
  int32_t batch_num_;
  int32_t channel_num_;
};

template<DeviceType device_type, typename T, typename IDX>
struct AvgPoolKernelUtil {
  static void Avgpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d);

  static void Avgpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d);

  static void Avgpool2dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d);

  static void Avgpool2dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d);

  static void Avgpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d);

  static void Avgpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d);
};

template<typename T, typename IDX>
OF_DEVICE_FUNC void Avgpool1dForwardCompute(const NdIndexOffsetHelper<IDX, 2> index_helper,
                                            IDX elem_num, const T* src, T* dest,
                                            const int32_t padding_l, const int32_t n_batch,
                                            const int32_t n_channel, const int32_t x_length,
                                            const int32_t kernel_size_l, const int32_t stride_l,
                                            const bool count_include_pad,
                                            const int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, l;
    index_helper.OffsetToNdIndex(num, n_c, l);

    const IDX start_idx = n_c * x_length;
    IDX lstart = l * stride_l - padding_l;
    IDX lend = XPU_INT_MIN<IDX>(lstart + kernel_size_l, x_length + padding_l);
    const IDX pool_size = (lend - lstart);

    lstart = XPU_INT_MAX<IDX>(0, lstart);
    lend = XPU_INT_MIN<IDX>(lend, x_length);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (lend - lstart);
      }
    }
    T sum = 0;

    const T* data = src + start_idx;
    for (IDX idx = lstart; idx < lend; idx += 1) { sum += data[idx]; }
    dest[num] = static_cast<T>(sum / divide_factor);
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Avgpool1dBackwardCompute(const NdIndexOffsetHelper<IDX, 2> index_helper,
                                             IDX elem_num, const T* src, T* dest,
                                             const int32_t padding_l, const int32_t n_batch,
                                             const int32_t n_channel, const int32_t input_length,
                                             const int32_t kernel_size_l, const int32_t stride_l,
                                             const bool count_include_pad,
                                             const int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, l;
    index_helper.OffsetToNdIndex(num, n_c, l);

    const IDX start_idx = n_c * input_length;
    IDX lstart = l * stride_l - padding_l;
    IDX lend = XPU_INT_MIN<IDX>(lstart + kernel_size_l, input_length + padding_l);
    const IDX pool_size = (lend - lstart);

    lstart = XPU_INT_MAX<IDX>(IDX(0), lstart);
    lend = XPU_INT_MIN<IDX>(lend, input_length);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (lend - lstart);
      }
    }
    T grad_delta = src[num] / divide_factor;
    T* data = dest + start_idx;
    for (IDX idx = lstart; idx < lend; idx += 1) {
      XPUAdd<T>::Invoke(&grad_delta, &data[idx]);  // dest[search_idx] += grad_delta
    }
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Avgpool2dForwardCompute(
    const NdIndexOffsetHelper<IDX, 3> index_helper, int64_t elem_num, const T* src, T* dest,
    const int32_t padding_h, const int32_t padding_w, const int32_t n_batch,
    const int32_t n_channel, const int32_t x_height, const int32_t x_width,
    const int32_t kernel_size_h, const int32_t kernel_size_w, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, h, w;
    index_helper.OffsetToNdIndex(num, n_c, h, w);

    const IDX start_idx = n_c * x_width * x_height;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;

    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, x_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, x_width + padding_w);
    const IDX pool_size = (hend - hstart) * (wend - wstart);

    hstart = XPU_INT_MAX<IDX>(0, hstart);
    wstart = XPU_INT_MAX<IDX>(0, wstart);
    hend = XPU_INT_MIN<IDX>(hend, x_height);
    wend = XPU_INT_MIN<IDX>(wend, x_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    T sum = 0;

    const T* data = src + start_idx;
    for (int64_t i = hstart; i < hend; i += 1) {
      for (int64_t j = wstart; j < wend; j += 1) {
        const IDX window_idx = i * x_width + j;
        sum += data[window_idx];
      }
    }
    dest[num] = sum / divide_factor;
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Avgpool2dBackwardCompute(
    const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num, const T* src, T* dest,
    const int32_t padding_h, const int32_t padding_w, const int32_t n_batch,
    const int32_t n_channel, const int32_t input_height, const int32_t input_width,
    const int32_t kernel_size_h, const int32_t kernel_size_w, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, h, w;
    index_helper.OffsetToNdIndex(num, n_c, h, w);

    const IDX start_idx = n_c * input_width * input_height;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, input_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, input_width + padding_w);
    const IDX pool_size = (hend - hstart) * (wend - wstart);

    hstart = XPU_INT_MAX<IDX>(IDX(0), hstart);
    wstart = XPU_INT_MAX<IDX>(IDX(0), wstart);
    hend = XPU_INT_MIN<IDX>(hend, input_height);
    wend = XPU_INT_MIN<IDX>(wend, input_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    T grad_delta = src[num] / divide_factor;
    T* data = dest + start_idx;
    for (IDX i = hstart; i < hend; i += 1) {
      for (IDX j = wstart; j < wend; j += 1) {
        const IDX window_idx = i * input_width + j;
        XPUAdd<T>::Invoke(&grad_delta, &data[window_idx]);  // dest[search_idx] += grad_delta
      }
    }
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Avgpool3dForwardCompute(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const T* src, T* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, t, h, w;
    index_helper.OffsetToNdIndex(num, n_c, t, h, w);

    const IDX start_idx = n_c * x_time * x_height * x_width;
    IDX tstart = t * stride_t - padding_t;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    IDX tend = XPU_INT_MIN<IDX>(tstart + kernel_size_t, x_time + padding_t);
    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, x_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, x_width + padding_w);
    const IDX pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

    tstart = XPU_INT_MAX<IDX>(IDX(0), tstart);
    hstart = XPU_INT_MAX<IDX>(IDX(0), hstart);
    wstart = XPU_INT_MAX<IDX>(IDX(0), wstart);
    tend = XPU_INT_MIN<IDX>(tend, x_time);
    hend = XPU_INT_MIN<IDX>(hend, x_height);
    wend = XPU_INT_MIN<IDX>(wend, x_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
      }
    }
    T sum = 0;
    const T* data = src + start_idx;
    for (IDX i = tstart; i < tend; i += 1) {
      for (IDX j = hstart; j < hend; j += 1) {
        for (IDX k = wstart; k < wend; k += 1) {
          const IDX window_idx = i * x_height * x_width + j * x_width + k;
          sum += data[window_idx];
        }
      }
    }
    dest[num] = sum / divide_factor;
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Avgpool3dBackwardCompute(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const T* src, T* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, const int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, t, h, w;
    index_helper.OffsetToNdIndex(num, n_c, t, h, w);

    const IDX start_idx = n_c * x_time * x_width * x_height;
    IDX tstart = t * stride_t - padding_t;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    IDX tend = XPU_INT_MIN<IDX>(tstart + kernel_size_t, x_time + padding_t);
    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, x_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, x_width + padding_w);
    const IDX pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

    tstart = XPU_INT_MAX<IDX>(IDX(0), tstart);
    hstart = XPU_INT_MAX<IDX>(IDX(0), hstart);
    wstart = XPU_INT_MAX<IDX>(IDX(0), wstart);
    tend = XPU_INT_MIN<IDX>(tend, x_time);
    hend = XPU_INT_MIN<IDX>(hend, x_height);
    wend = XPU_INT_MIN<IDX>(wend, x_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
      }
    }
    T grad_delta = src[num] / divide_factor;
    T* data = dest + start_idx;
    for (IDX i = tstart; i < tend; i += 1) {
      for (IDX j = hstart; j < hend; j += 1) {
        for (IDX k = wstart; k < wend; k += 1) {
          const IDX window_idx = i * x_height * x_width + j * x_width + k;
          XPUAdd<T>::Invoke(&grad_delta, &data[window_idx]);  // dest[search_idx] += grad_delta
        }
      }
    }
  }
}

#ifdef WITH_CUDA
template<DeviceType device_type, typename IDX>
struct AvgPoolKernelUtil<device_type, half, IDX> {
  static void Avgpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const half* src, half* dest,
                               const AvgPoolParams3D& params_3d);

  static void Avgpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const half* src, half* dest,
                                const AvgPoolParams3D& params_3d);

  static void Avgpool2dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                               const IDX elem_num, const half* src, half* dest,
                               const AvgPoolParams3D& params_3d);

  static void Avgpool2dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                const IDX elem_num, const half* src, half* dest,
                                const AvgPoolParams3D& params_3d);

  static void Avgpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const half* src, half* dest,
                               const AvgPoolParams3D& params_3d);

  static void Avgpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                const IDX elem_num, const half* src, half* dest,
                                const AvgPoolParams3D& params_3d);
};

template<typename IDX>
OF_DEVICE_FUNC void HalfAvgpool1dForwardCompute(const NdIndexOffsetHelper<IDX, 2> index_helper,
                                                IDX elem_num, const half* src, half* dest,
                                                const int32_t padding_l, const int32_t n_batch,
                                                const int32_t n_channel, const int32_t x_length,
                                                const int32_t kernel_size_l, const int32_t stride_l,
                                                const bool count_include_pad,
                                                const int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, l;
    index_helper.OffsetToNdIndex(num, n_c, l);

    const IDX start_idx = n_c * x_length;
    IDX lstart = l * stride_l - padding_l;
    IDX lend = XPU_INT_MIN<IDX>(lstart + kernel_size_l, x_length + padding_l);
    const IDX pool_size = (lend - lstart);

    lstart = XPU_INT_MAX<IDX>(0, lstart);
    lend = XPU_INT_MIN<IDX>(lend, x_length);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (lend - lstart);
      }
    }
    float sum = 0;

    const half* data = src + start_idx;
    for (IDX idx = lstart; idx < lend; idx += 1) { sum += __half2float(data[idx]); }
    dest[num] = __float2half(sum / divide_factor);
  }
}

template<typename IDX>
OF_DEVICE_FUNC void HalfAvgpool1dBackwardCompute(
    const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num, const half* src, half* dest,
    const int32_t padding_l, const int32_t n_batch, const int32_t n_channel,
    const int32_t input_length, const int32_t kernel_size_l, const int32_t stride_l,
    const bool count_include_pad, const int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, l;
    index_helper.OffsetToNdIndex(num, n_c, l);

    const IDX start_idx = n_c * input_length;
    IDX lstart = l * stride_l - padding_l;
    IDX lend = XPU_INT_MIN<IDX>(lstart + kernel_size_l, input_length + padding_l);
    const IDX pool_size = (lend - lstart);

    lstart = XPU_INT_MAX<IDX>(IDX(0), lstart);
    lend = XPU_INT_MIN<IDX>(lend, input_length);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (lend - lstart);
      }
    }
    half grad_delta = static_cast<half>(__half2float(src[num]) / divide_factor);
    half* data = dest + start_idx;
    for (IDX idx = lstart; idx < lend; idx += 1) { XPUAdd<half>::Invoke(&grad_delta, &data[idx]); }
  }
}

template<typename IDX>
OF_DEVICE_FUNC void HalfAvgpool2dForwardCompute(
    const NdIndexOffsetHelper<IDX, 3> index_helper, int64_t elem_num, const half* src, half* dest,
    const int32_t padding_h, const int32_t padding_w, const int32_t n_batch,
    const int32_t n_channel, const int32_t x_height, const int32_t x_width,
    const int32_t kernel_size_h, const int32_t kernel_size_w, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, h, w;
    index_helper.OffsetToNdIndex(num, n_c, h, w);

    const IDX start_idx = n_c * x_width * x_height;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;

    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, x_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, x_width + padding_w);
    const IDX pool_size = (hend - hstart) * (wend - wstart);

    hstart = XPU_INT_MAX<IDX>(0, hstart);
    wstart = XPU_INT_MAX<IDX>(0, wstart);
    hend = XPU_INT_MIN<IDX>(hend, x_height);
    wend = XPU_INT_MIN<IDX>(wend, x_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    float sum = 0;
    const half* data = src + start_idx;
    for (int64_t i = hstart; i < hend; i += 1) {
      for (int64_t j = wstart; j < wend; j += 1) {
        const IDX window_idx = i * x_width + j;
        sum += __half2float(data[window_idx]);
      }
    }
    dest[num] = __float2half(sum / divide_factor);
  }
}

template<typename IDX>
OF_DEVICE_FUNC void HalfAvgpool2dBackwardCompute(
    const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num, const half* src, half* dest,
    const int32_t padding_h, const int32_t padding_w, const int32_t n_batch,
    const int32_t n_channel, const int32_t input_height, const int32_t input_width,
    const int32_t kernel_size_h, const int32_t kernel_size_w, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, h, w;
    index_helper.OffsetToNdIndex(num, n_c, h, w);

    const IDX start_idx = n_c * input_width * input_height;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, input_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, input_width + padding_w);
    const IDX pool_size = (hend - hstart) * (wend - wstart);

    hstart = XPU_INT_MAX<IDX>(IDX(0), hstart);
    wstart = XPU_INT_MAX<IDX>(IDX(0), wstart);
    hend = XPU_INT_MIN<IDX>(hend, input_height);
    wend = XPU_INT_MIN<IDX>(wend, input_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    half grad_delta = static_cast<half>(__half2float(src[num]) / divide_factor);
    half* data = dest + start_idx;
    for (IDX i = hstart; i < hend; i += 1) {
      for (IDX j = wstart; j < wend; j += 1) {
        const IDX window_idx = i * input_width + j;
        XPUAdd<half>::Invoke(&grad_delta, &data[window_idx]);
      }
    }
  }
}

template<typename IDX>
OF_DEVICE_FUNC void HalfAvgpool3dForwardCompute(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const half* src, half* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, t, h, w;
    index_helper.OffsetToNdIndex(num, n_c, t, h, w);

    const IDX start_idx = n_c * x_time * x_height * x_width;
    IDX tstart = t * stride_t - padding_t;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    IDX tend = XPU_INT_MIN<IDX>(tstart + kernel_size_t, x_time + padding_t);
    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, x_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, x_width + padding_w);
    const IDX pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

    tstart = XPU_INT_MAX<IDX>(IDX(0), tstart);
    hstart = XPU_INT_MAX<IDX>(IDX(0), hstart);
    wstart = XPU_INT_MAX<IDX>(IDX(0), wstart);
    tend = XPU_INT_MIN<IDX>(tend, x_time);
    hend = XPU_INT_MIN<IDX>(hend, x_height);
    wend = XPU_INT_MIN<IDX>(wend, x_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
      }
    }
    float sum = 0;
    const half* data = src + start_idx;
    for (IDX i = tstart; i < tend; i += 1) {
      for (IDX j = hstart; j < hend; j += 1) {
        for (IDX k = wstart; k < wend; k += 1) {
          const IDX window_idx = i * x_height * x_width + j * x_width + k;
          sum += __half2float(data[window_idx]);
        }
      }
    }
    dest[num] = __float2half(sum / divide_factor);
  }
}

template<typename IDX>
OF_DEVICE_FUNC void HalfAvgpool3dBackwardCompute(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const half* src, half* dest,
    const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const bool count_include_pad, const int32_t divisor_override) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, t, h, w;
    index_helper.OffsetToNdIndex(num, n_c, t, h, w);

    const IDX start_idx = n_c * x_time * x_width * x_height;
    IDX tstart = t * stride_t - padding_t;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    IDX tend = XPU_INT_MIN<IDX>(tstart + kernel_size_t, x_time + padding_t);
    IDX hend = XPU_INT_MIN<IDX>(hstart + kernel_size_h, x_height + padding_h);
    IDX wend = XPU_INT_MIN<IDX>(wstart + kernel_size_w, x_width + padding_w);
    const IDX pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

    tstart = XPU_INT_MAX<IDX>(IDX(0), tstart);
    hstart = XPU_INT_MAX<IDX>(IDX(0), hstart);
    wstart = XPU_INT_MAX<IDX>(IDX(0), wstart);
    tend = XPU_INT_MIN<IDX>(tend, x_time);
    hend = XPU_INT_MIN<IDX>(hend, x_height);
    wend = XPU_INT_MIN<IDX>(wend, x_width);

    IDX divide_factor;
    if (divisor_override != static_cast<int32_t>(0)) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
      }
    }
    half grad_delta = static_cast<half>(__half2float(src[num]) / divide_factor);
    half* data = dest + start_idx;
    for (IDX i = tstart; i < tend; i += 1) {
      for (IDX j = hstart; j < hend; j += 1) {
        for (IDX k = wstart; k < wend; k += 1) {
          const IDX window_idx = i * x_height * x_width + j * x_width + k;
          XPUAdd<half>::Invoke(&grad_delta, &data[window_idx]);
        }
      }
    }
  }
}

#endif  // WITH_CUDA

#define INSTANTIATE_AVG_POOL_KERNEL_UTIL(device_type_v, dtype_pair, index_dtype_pair) \
  template struct AvgPoolKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),      \
                                    OF_PP_PAIR_FIRST(index_dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_AVG_POOL_KERNEL_UTIL_H_
