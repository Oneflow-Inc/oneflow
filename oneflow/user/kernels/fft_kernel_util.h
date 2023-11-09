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
#ifndef ONEFLOW_USER_KERNELS_FFT_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_FFT_KERNEL_UTIL_H_

#include <cstdint>
#include <type_traits>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<typename T>
inline T _fft_normalization_scale(const int32_t frame_length, bool normalized) {
  if (!normalized) { return static_cast<T>(1.0); }
  return static_cast<T>(1.0 / std::sqrt(frame_length));
}

template<DeviceType device_type, typename T>
struct FillConjSymmetryUtil {
  static void FillConjSymmetryForward(ep::Stream* stream, T* data_out, const Shape& shape,
                                      const Stride& strides, const int64_t last_dim,
                                      int64_t elem_count);
};

template<DeviceType device_type, typename real_type, typename complex_type>
struct ComplexConvertUtil {
  static void ConvertToDoubleSized(ep::Stream* stream, const complex_type* in, complex_type* dst,
                                   size_t len, size_t n);
  static void ConvertComplexToReal(ep::Stream* stream, const complex_type* in, real_type* out,
                                   size_t n);
};

template<DeviceType device_type, typename T, typename FCT_TYPE>
struct FftC2CKernelUtil {
  static void FftC2CForward(ep::Stream* stream, const T* data_in, T* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, FCT_TYPE norm_fct,
                            DataType real_type);
};

template<DeviceType device_type, typename IN, typename OUT>
struct FftR2CKernelUtil {
  static void FftR2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, IN norm_fct, DataType real_type);
};

template<DeviceType device_type, typename IN, typename OUT>
struct FftC2RKernelUtil {
  static void FftC2RForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            int64_t last_dim_size, const std::vector<int64_t>& dims, OUT norm_fct,
                            DataType real_type);
};

template<DeviceType device_type, typename IN, typename OUT>
struct FftStftKernelUtil {
  static void FftStftForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                             const Shape& input_shape, const Shape& output_shape,
                             const Stride& input_stride, const Stride& output_stride, bool forward,
                             const std::vector<int64_t>& axes, IN norm_fct, int64_t len,
                             int64_t dims, int64_t batch);
};

}  // namespace oneflow
#endif  // ONEFLOW_USER_KERNELS_FFT_KERNEL_UTIL_H_