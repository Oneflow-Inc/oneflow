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
#ifndef ONEFLOW_USER_KERNELS_POOL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_POOL_KERNEL_UTIL_H_
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

#define POOL_DATA_TYPE_SEQ                        \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)   \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define POOL_IDX_DATA_TYPE_SEQ                    \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define POOL_DATA_TYPE_CPU_SEQ POOL_DATA_TYPE_SEQ
#define POOL_DATA_TYPE_CUDA_SEQ POOL_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(half, DataType::kFloat16)

typedef small_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;

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

class MaxPoolParams3D {
 public:
  MaxPoolParams3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                  const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size,
                  const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation,
                  const bool return_indices, const bool ceil_mode);
  ~MaxPoolParams3D() = default;

  const std::string& data_format() const { return data_format_; }
  const std::vector<int32_t>& padding() const { return padding_; }
  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& stride_3d() const { return stride_3d_; }
  const std::vector<int32_t>& dilation_3d() const { return dilation_3d_; }
  const bool& return_indices() const { return return_indices_; }
  const bool& ceil_mode() const { return ceil_mode_; }
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
  std::vector<int32_t> dilation_3d_;
  bool return_indices_;
  bool ceil_mode_;
  int32_t batch_num_;
  int32_t channel_num_;
};

template<DeviceType device_type, typename T, typename IDX>
struct PoolKernelUtil {
  static void Maxpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolParams3D& params_3d);

  static void Maxpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolParams3D& params_3d);

  static void Maxpool2dForwardCFirst(ep::Stream* stream,
                                     const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                     const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                     const MaxPoolParams3D& params_3d);

  static void Maxpool2dBackwardCFirst(ep::Stream* stream,
                                      const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                      const IDX elem_num, const T* src, T* dest,
                                      const int64_t* indice_ptr, const MaxPoolParams3D& params_3d);

  static void Maxpool2dForwardCLast(ep::Stream* stream,
                                    const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                    const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                    const MaxPoolParams3D& params_3d);

  static void Maxpool2dBackwardCLast(ep::Stream* stream,
                                     const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                     const IDX elem_num, const T* src, T* dest,
                                     const int64_t* indice_ptr, const MaxPoolParams3D& params_3d);

  static void Maxpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolParams3D& params_3d);

  static void Maxpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolParams3D& params_3d);
};

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool1dForwardCompute(const NdIndexOffsetHelper<IDX, 2> index_helper,
                                            IDX elem_num, const T* src, T* dest,
                                            int64_t* indice_ptr, const int32_t padding_l,
                                            const int32_t n_batch, const int32_t n_channel,
                                            const int32_t x_length, const int32_t kernel_size_l,
                                            const int32_t stride_l, const int32_t dilation_l) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, l;
    index_helper.OffsetToNdIndex(num, n_c, l);

    IDX lstart = l * stride_l - padding_l;
    const IDX lend = (lstart + (kernel_size_l - 1) * dilation_l + 1) <= x_length
                         ? (lstart + (kernel_size_l - 1) * dilation_l + 1)
                         : x_length;

    while (lstart < 0) { lstart += dilation_l; }

    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    IDX max_index = lstart;

    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();
    const T* data = src + n_c * x_length;
    for (IDX idx = lstart; idx < lend; idx += dilation_l) {
      const IDX window_idx = idx;
      T val = data[window_idx];
      if (val > max_value || detail::numerics<T>::isnan(val)) {
        max_value = val;
        max_index = idx;
      }
    }
    dest[num] = max_value;
    indice_ptr[num] = max_index;
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool1dBackwardCompute(const NdIndexOffsetHelper<IDX, 2> index_helper,
                                             const IDX elem_num, const T* src, T* dest,
                                             const int64_t* indice_ptr, const int32_t n_batch,
                                             const int32_t n_channel, const int32_t src_length,
                                             const int32_t dst_length) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, l;
    index_helper.OffsetToNdIndex(num, n_c, l);

    const IDX src_start = n_c * src_length;
    const IDX dst_start = n_c * dst_length;
    const IDX index = src_start + l;
    const IDX max_index = dst_start + indice_ptr[index];
    if (max_index != -1) {
      /* update gradient, equals to dest[max_index] += src[index]; */
      DeviceAdd<T>::Invoke(src + index, dest + max_index);
    }
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool2dForwardComputeCFirst(
    const NdIndexOffsetHelper<IDX, 3> index_helper, IDX elem_num, const T* src, T* dest,
    int64_t* indice_ptr, const int32_t padding_h, const int32_t padding_w, const int32_t n_batch,
    const int32_t n_channel, const int32_t x_height, const int32_t x_width,
    const int32_t kernel_size_h, const int32_t kernel_size_w, const int32_t stride_h,
    const int32_t stride_w, const int32_t dilation_h, const int32_t dilation_w) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, h, w;
    index_helper.OffsetToNdIndex(num, n_c, h, w);
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    const IDX hend = (hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height
                         ? (hstart + (kernel_size_h - 1) * dilation_h + 1)
                         : x_height;
    const IDX wend = (wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width
                         ? (wstart + (kernel_size_w - 1) * dilation_w + 1)
                         : x_width;
    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }
    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();
    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    IDX max_index = hstart * x_width + wstart;
    const T* data = src + n_c * x_width * x_height;
    for (IDX i = hstart; i < hend; i += dilation_h) {
      for (IDX j = wstart; j < wend; j += dilation_w) {
        const IDX window_idx = i * x_width + j;
        T val = data[window_idx];
        /* NOTE:
        std::isnan(val) only supports a few data types, see:
        https://en.cppreference.com/w/cpp/numeric/math/isnan and when use gcc/g++ 4.x to compile,
        the following exception will be throw:

        new_kernel_util.cu:24] Check failed: cudaMemcpyAsync(dst, src, sz, cudaMemcpyDefault,
        ctx->cuda_stream() ) : unspecified launch failure (719)

        but if use gcc/g++ 7.x to compile, everything is ok! the exact reason is still unknown!
        */
        if (val > max_value || detail::numerics<T>::isnan(val)) {
          max_index = window_idx;
          max_value = val;
        }
      }
    }
    dest[num] = max_value;
    indice_ptr[num] = max_index;
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool2dBackwardComputeCFirst(
    const NdIndexOffsetHelper<IDX, 3> index_helper, const IDX elem_num, const T* src, T* dest,
    const int64_t* indice_ptr, const int32_t n_batch, const int32_t n_channel,
    const int32_t src_height, const int32_t src_width, const int32_t dst_height,
    const int32_t dst_width) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, h, w;
    index_helper.OffsetToNdIndex(num, n_c, h, w);

    const IDX src_start = n_c * src_height * src_width;
    const IDX dst_start = n_c * dst_height * dst_width;
    const IDX index = src_start + h * src_width + w;

    const IDX max_index = dst_start + indice_ptr[index];
    if (max_index != -1) {
      /* update gradient, equals to dest[max_index] += src[index]; */
      DeviceAdd<T>::Invoke(src + index, dest + max_index);
    }
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool2dBackwardComputeCLast(const NdIndexOffsetHelper<IDX, 4> index_helper,
                                                  const IDX elem_num, const T* src, T* dest,
                                                  const int64_t* indice_ptr, const int32_t n_batch,
                                                  const int32_t n_channel, const int32_t src_height,
                                                  const int32_t src_width, const int32_t dst_height,
                                                  const int32_t dst_width) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n, c, h, w;
    index_helper.OffsetToNdIndex(num, n, c, h, w);
    const IDX src_start = n * src_height * src_width * n_channel;
    const IDX dst_start = n * dst_height * dst_width * n_channel;
    const IDX index = src_start + h * src_width + w;
    const IDX max_index = dst_start + indice_ptr[index];
    if (max_index != -1) {
      /* update gradient, equals to dest[max_index] += src[index]; */
      DeviceAdd<T>::Invoke(src + index, dest + max_index);
    }
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool3dForwardCompute(
    const NdIndexOffsetHelper<IDX, 4> index_helper, IDX elem_num, const T* src, T* dest,
    int64_t* indice_ptr, const int32_t padding_t, const int32_t padding_h, const int32_t padding_w,
    const int32_t n_batch, const int32_t n_channel, const int32_t x_time, const int32_t x_height,
    const int32_t x_width, const int32_t kernel_size_t, const int32_t kernel_size_h,
    const int32_t kernel_size_w, const int32_t stride_t, const int32_t stride_h,
    const int32_t stride_w, const int32_t dilation_t, const int32_t dilation_h,
    const int32_t dilation_w) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, t, h, w;
    index_helper.OffsetToNdIndex(num, n_c, t, h, w);

    IDX tstart = t * stride_t - padding_t;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;

    const IDX t1 = tstart + (kernel_size_t - 1) * dilation_t + 1;
    const IDX t2 = hstart + (kernel_size_h - 1) * dilation_h + 1;
    const IDX t3 = wstart + (kernel_size_w - 1) * dilation_w + 1;
    const IDX tend = t1 <= x_time ? t1 : x_time;
    const IDX hend = t2 <= x_height ? t2 : x_height;
    const IDX wend = t3 <= x_width ? t3 : x_width;

    while (tstart < 0) { tstart += dilation_t; }
    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }

    IDX max_index = tstart * x_height * x_width + hstart * x_width + wstart;
    const T* data = src + n_c * x_time * x_width * x_height;
    T max_value = detail::numeric_limits<T>::lower_bound();
    for (IDX zi = tstart; zi < tend; zi += dilation_t) {
      for (IDX i = hstart; i < hend; i += dilation_h) {
        for (IDX j = wstart; j < wend; j += dilation_w) {
          const IDX window_idx = zi * x_height * x_width + i * x_width + j;
          T val = data[window_idx];
          if (val > max_value || detail::numerics<T>::isnan(val)) {
            max_value = val;
            max_index = window_idx;
          }
        }
      }
      /* set output to local max */
      dest[num] = max_value;
      /* store location of max */
      indice_ptr[num] = max_index;
    }
  }
}

template<typename T, typename IDX>
OF_DEVICE_FUNC void Maxpool3dBackwardCompute(const NdIndexOffsetHelper<IDX, 4> index_helper,
                                             const IDX elem_num, const T* src, T* dest,
                                             const int64_t* indice_ptr, const int32_t n_batch,
                                             const int32_t n_channel, const int32_t src_time,
                                             const int32_t src_height, const int32_t src_width,
                                             const int32_t dst_time, const int32_t dst_height,
                                             const int32_t dst_width) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX n_c, t, h, w;
    index_helper.OffsetToNdIndex(num, n_c, t, h, w);

    const IDX src_start = n_c * src_time * src_height * src_width;
    const IDX dst_start = n_c * dst_time * dst_height * dst_width;
    const IDX index = src_start + t * src_height * src_width + h * src_width + w;
    const IDX max_index = dst_start + indice_ptr[index];

    if (max_index != -1) { DeviceAdd<T>::Invoke(src + index, dest + max_index); }
  }
}

#define INSTANTIATE_POOL_KERNEL_UTIL(device_type_v, dtype_pair, index_dtype_pair) \
  template struct PoolKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),     \
                                 OF_PP_PAIR_FIRST(index_dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_POOL_KERNEL_UTIL_H_
