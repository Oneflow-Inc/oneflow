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

#define POOLING_DATA_TYPE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)   \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define POOLING_DATA_TYPE_CPU_SEQ POOLING_DATA_TYPE_SEQ

#define POOLING_DATA_TYPE_CUDA_SEQ POOLING_DATA_TYPE_SEQ

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;

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

class MaxPoolingParams3D {
 public:
  MaxPoolingParams3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                     const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size,
                     const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation,
                     const bool return_indices, const bool ceil_mode);
  ~MaxPoolingParams3D() = default;

  const std::string& data_format() const { return data_format_; }
  const std::vector<int32_t>& padding() const { return padding_; }
  const std::vector<int32_t>& pooling_size_3d() const { return pooling_size_3d_; }
  const std::vector<int32_t>& stride_3d() const { return stride_3d_; }
  const std::vector<int32_t>& dilation_3d() const { return dilation_3d_; }
  const bool& return_indices() const { return return_indices_; }
  const bool& ceil_mode() const { return ceil_mode_; }
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
  std::vector<int32_t> dilation_3d_;
  bool return_indices_;
  bool ceil_mode_;
  int64_t batch_num_;
  int64_t channel_num_;
};

template<DeviceType device_type, typename T>
struct PoolingKernelUtil {
  static void Maxpool1dForward(ep::Stream* stream,
                               const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolingParams3D& params_3d);

  static void Maxpool1dBackward(ep::Stream* stream,
                                const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolingParams3D& params_3d);

  static void Maxpool2dForwardCFirst(ep::Stream* stream,
                                     const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                     const int64_t elem_num, const T* src, T* dest,
                                     int64_t* indice_ptr, const MaxPoolingParams3D& params_3d);

  static void Maxpool2dBackwardCFirst(ep::Stream* stream,
                                      const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                      const int64_t elem_num, const T* src, T* dest,
                                      const int64_t* indice_ptr,
                                      const MaxPoolingParams3D& params_3d);

  static void Maxpool2dForwardCLast(ep::Stream* stream,
                                    const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                    const int64_t elem_num, const T* src, T* dest,
                                    int64_t* indice_ptr, const MaxPoolingParams3D& params_3d);

  static void Maxpool2dBackwardCLast(ep::Stream* stream,
                                     const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                     const int64_t elem_num, const T* src, T* dest,
                                     const int64_t* indice_ptr,
                                     const MaxPoolingParams3D& params_3d);

  static void Maxpool3dForward(ep::Stream* stream,
                               const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolingParams3D& params_3d);

  static void Maxpool3dBackward(ep::Stream* stream,
                                const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolingParams3D& params_3d);
};

template<typename T>
OF_DEVICE_FUNC void Maxpool1dForwardCompute(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                            int64_t elem_num, const T* src, T* dest,
                                            int64_t* indice_ptr, const int32_t padding_l,
                                            const int64_t n_batch, const int64_t n_channel,
                                            const int64_t x_length, const int64_t y_length,
                                            const int32_t kernel_size_l, const int32_t stride_l,
                                            const int32_t dilation_l) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, l;
    index_helper.OffsetToNdIndex(num, n, c, l);

    // n, c, l->index = n*c*l + c* l
    const int64_t start_idx = (n * n_channel + c) * x_length;
    int64_t lstart = l * stride_l - padding_l;
    const int64_t lend = (lstart + (kernel_size_l - 1) * dilation_l + 1) <= x_length
                             ? (lstart + (kernel_size_l - 1) * dilation_l + 1)
                             : x_length;

    while (lstart < 0) { lstart += dilation_l; }

    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    int64_t max_index = lstart;
    int64_t src_idx = 0;

    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();

    for (int64_t idx = lstart; idx < lend; idx += dilation_l) {
      const int64_t search_idx = start_idx + idx;
      T val = src[search_idx];
      if (val > max_value || detail::numerics<T>::isnan(val)) {
        max_value = val;
        max_index = idx;
        src_idx = search_idx;
      }
    }
    dest[num] = src[src_idx];
    indice_ptr[num] = max_index;
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool1dBackwardCompute(const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                             const int64_t elem_num, const T* src, T* dest,
                                             const int64_t* indice_ptr, const int64_t n_batch,
                                             const int64_t n_channel, const int64_t src_length,
                                             const int64_t dst_length) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, l;
    index_helper.OffsetToNdIndex(num, n, c, l);

    const int64_t src_start = (n * n_channel + c) * src_length;
    const int64_t dst_start = (n * n_channel + c) * dst_length;
    const int64_t index = src_start + l;
    const int64_t max_index = dst_start + indice_ptr[index];
    if (max_index != -1) {
      /* update gradient, equals to dest[max_index] += src[index]; */
      DeviceAdd<T>::Invoke(src + index, dest + max_index);
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool2dForwardComputeCFirst(
    const NdIndexOffsetHelper<int64_t, 4> index_helper, int64_t elem_num, const T* src, T* dest,
    int64_t* indice_ptr, const int32_t padding_h, const int32_t padding_w, const int64_t n_batch,
    const int64_t n_channel, const int64_t x_height, const int64_t x_width, const int64_t y_height,
    const int64_t y_width, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_h, const int32_t stride_w, const int32_t dilation_h,
    const int32_t dilation_w) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w;
    index_helper.OffsetToNdIndex(num, n, c, h, w);

    const int64_t start_idx = (n * n_channel + c) * x_width * x_height;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    const int64_t hend = (hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height
                             ? (hstart + (kernel_size_h - 1) * dilation_h + 1)
                             : x_height;
    const int64_t wend = (wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width
                             ? (wstart + (kernel_size_w - 1) * dilation_w + 1)
                             : x_width;

    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }

    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    int64_t max_index = hstart * x_width + wstart;
    int64_t src_idx = 0;

    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();

    for (int64_t i = hstart; i < hend; i += dilation_h) {
      for (int64_t j = wstart; j < wend; j += dilation_w) {
        const int64_t window_idx = i * x_width + j;
        const int64_t search_idx = start_idx + window_idx;
        T val = src[search_idx];
        /* NOTE:
        std::isnan(val) only supports a few data types, see:
        https://en.cppreference.com/w/cpp/numeric/math/isnan and when use gcc/g++ 4.x to compile,
        the following exception will be throw:

        new_kernel_util.cu:24] Check failed: cudaMemcpyAsync(dst, src, sz, cudaMemcpyDefault,
        ctx->cuda_stream() ) : unspecified launch failure (719)

        but if use gcc/g++ 7.x to compile, everything is ok! the exact reason is still unknown!
        */
        if (val > max_value || detail::numerics<T>::isnan(val)) {
          max_value = val;
          max_index = window_idx;
          src_idx = search_idx;
        }
      }
    }
    dest[num] = src[src_idx];
    indice_ptr[num] = max_index;
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool2dBackwardComputeCFirst(
    const NdIndexOffsetHelper<int64_t, 4> index_helper, const int64_t elem_num, const T* src,
    T* dest, const int64_t* indice_ptr, const int64_t n_batch, const int64_t n_channel,
    const int64_t src_height, const int64_t src_width, const int64_t dst_height,
    const int64_t dst_width) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w;
    index_helper.OffsetToNdIndex(num, n, c, h, w);

    const int64_t src_start = (n * n_channel + c) * src_height * src_width;
    const int64_t dst_start = (n * n_channel + c) * dst_height * dst_width;
    const int64_t index = src_start + h * src_width + w;

    const int64_t max_index = dst_start + indice_ptr[index];
    if (max_index != -1) {
      /* update gradient, equals to dest[max_index] += src[index]; */
      DeviceAdd<T>::Invoke(src + index, dest + max_index);
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool2dBackwardComputeCLast(
    const NdIndexOffsetHelper<int64_t, 4> index_helper, const int64_t elem_num, const T* src,
    T* dest, const int64_t* indice_ptr, const int64_t n_batch, const int64_t n_channel,
    const int64_t src_height, const int64_t src_width, const int64_t dst_height,
    const int64_t dst_width) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, h, w;
    index_helper.OffsetToNdIndex(num, n, c, h, w);
    const int64_t src_start = n * src_height * src_width * n_channel;
    const int64_t dst_start = n * dst_height * dst_width * n_channel;
    const int64_t index = src_start + h * src_width + w;
    const int64_t max_index = dst_start + indice_ptr[index];
    if (max_index != -1) {
      /* update gradient, equals to dest[max_index] += src[index]; */
      DeviceAdd<T>::Invoke(src + index, dest + max_index);
    }
  }
}

template<typename T>
OF_DEVICE_FUNC void Maxpool3dForwardCompute(
    const NdIndexOffsetHelper<int64_t, 5> index_helper, int64_t elem_num, const T* src, T* dest,
    int64_t* indice_ptr, const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int64_t n_batch, const int64_t n_channel, const int64_t x_time, const int64_t x_height,
    const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
    const int32_t kernel_size_t, const int32_t kernel_size_h, const int32_t kernel_size_w,
    const int32_t stride_t, const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_t, const int32_t dilation_h, const int32_t dilation_w) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, t, h, w;
    index_helper.OffsetToNdIndex(num, n, c, t, h, w);

    int64_t xstart = n * n_channel * x_time * x_width * x_height;
    int64_t start_idx = xstart + c * x_time * x_width * x_height;
    int64_t tstart = t * stride_t - padding_t;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;

    const int64_t t1 = tstart + (kernel_size_t - 1) * dilation_t + 1;
    const int64_t t2 = hstart + (kernel_size_h - 1) * dilation_h + 1;
    const int64_t t3 = wstart + (kernel_size_w - 1) * dilation_w + 1;
    const int64_t tend = t1 <= x_time ? t1 : x_time;
    const int64_t hend = t2 <= x_height ? t2 : x_height;
    const int64_t wend = t3 <= x_width ? t3 : x_width;

    while (tstart < 0) { tstart += dilation_t; }
    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }

    int64_t max_index = tstart * x_height * x_width + hstart * x_width + wstart;
    int64_t src_idx = 0;

    T max_value = detail::numeric_limits<T>::lower_bound();
    for (int64_t zi = tstart; zi < tend; zi += dilation_t) {
      for (int64_t i = hstart; i < hend; i += dilation_h) {
        for (int64_t j = wstart; j < wend; j += dilation_w) {
          const int64_t window_idx = zi * x_height * x_width + i * x_width + j;
          const int64_t search_idx = start_idx + window_idx;
          T val = src[search_idx];
          if (val > max_value || detail::numerics<T>::isnan(val)) {
            max_value = val;
            max_index = window_idx;
            src_idx = search_idx;
          }
        }
      }
      /* set output to local max */
      dest[num] = src[src_idx];
      /* store location of max */
      indice_ptr[num] = max_index;
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
                                             const int64_t dst_width) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    int64_t n, c, t, h, w;
    index_helper.OffsetToNdIndex(num, n, c, t, h, w);

    const int64_t src_start = (n * n_channel + c) * src_time * src_height * src_width;
    const int64_t dst_start = (n * n_channel + c) * dst_time * dst_height * dst_width;
    const int64_t index = src_start + t * src_height * src_width + h * src_width + w;
    const int64_t max_index = dst_start + indice_ptr[index];

    if (max_index != -1) { DeviceAdd<T>::Invoke(src + index, dest + max_index); }
  }
}

#define INSTANTIATE_POOLING_KERNEL_UTIL(device_type_v, dtype_pair) \
  template struct PoolingKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_POOLING_KERNEL_UTIL_H_
