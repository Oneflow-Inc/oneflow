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
#ifndef ONEFLOW_USER_KERNELS_UNPOOL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_UNPOOL_KERNEL_UTIL_H_
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

// #define POOL_DATA_TYPE_SEQ                        \
//   OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
//   OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)   \
//   OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

// #define POOL_IDX_DATA_TYPE_SEQ                    \
//   OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
//   OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

// #define POOL_DATA_TYPE_CPU_SEQ POOL_DATA_TYPE_SEQ
// #define POOL_DATA_TYPE_CUDA_SEQ POOL_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(half, DataType::kFloat16)

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

class MaxUnpoolParams3D {
 public:
  MaxUnpoolParams3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                    const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size,
                    const std::vector<int32_t>& stride);
  ~MaxUnpoolParams3D() = default;

  const std::string& data_format() const { return data_format_; }
  const std::vector<int32_t>& padding() const { return padding_; }
  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& stride_3d() const { return stride_3d_; }
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
  int32_t batch_num_;
  int32_t channel_num_;
};

template<typename T, typename IDX>
OF_DEVICE_FUNC void MaxUnpool1dForwardCompute(const NdIndexOffsetHelper<IDX, 2> index_helper,
                                            IDX elem_num, const T* src, T* dest,
                                            const int64_t* indice_ptr, const int32_t padding_l,
                                            const int32_t n_batch, const int32_t n_channel,
                                            const int32_t x_length, const int32_t kernel_size_l,
                                            const int32_t stride_l) {
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


// #define INSTANTIATE_POOL_KERNEL_UTIL(device_type_v, dtype_pair, index_dtype_pair) \
//   template struct PoolKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),     \
//                                  OF_PP_PAIR_FIRST(index_dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_POOL_KERNEL_UTIL_H_
