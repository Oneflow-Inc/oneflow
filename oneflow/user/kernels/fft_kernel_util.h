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
#include <vector>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

enum class fft_norm_mode {
  none = 0,   // No normalization
  by_root_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

// Convert NumPy compatible normalization mode string to enum values
// In Numpy, "forward" translates to `by_n` for a forward transform and `none` for backward.
fft_norm_mode norm_from_string(const Optional<std::string>& norm_op, bool forward) {
  std::string norm_str = norm_op.value_or("backward");
  if (norm_str == "backward") {
    return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  } else if (norm_str == "forward") {
    return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
  } else if (norm_str == "ortho") {
    return fft_norm_mode::by_root_n;
  }

  // if (norm_op){
  //     // std::string norm_str = *JUST(norm_op);
  //     if (*JUST(norm_op) == "backward"){
  //         return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  //     }
  //     else if (*JUST(norm_op) == "forward"){
  //         return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
  //     }
  //     else if (*JUST(norm_op) == "ortho"){
  //         return fft_norm_mode::by_root_n;
  //     }
  // }
  // else{
  //     return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  // }
  // CHECK_OR_RETURN(false) << "Invalid normalization mode: \"" << *JUST(norm_op) << "\"";
  return fft_norm_mode::none;
}

template<typename T>
T compute_fct(int64_t size, fft_norm_mode normalization) {
  constexpr auto one = static_cast<T>(1);
  switch (normalization) {
    case fft_norm_mode::none: return one;
    case fft_norm_mode::by_n: return one / static_cast<T>(size);
    case fft_norm_mode::by_root_n: return one / std::sqrt(static_cast<T>(size));
  }
  return static_cast<T>(0);
}

template<typename T>
T compute_fct(const Shape& in_shape, std::vector<int64_t> dims, fft_norm_mode normalization) {
  if (normalization == fft_norm_mode::none) { return static_cast<T>(1); }
  int64_t n = 1;
  for (int64_t idx : dims) { n *= in_shape.At(idx); }
  return compute_fct<T>(n, normalization);
}

template<typename T, int NDIM>
void _conj_symmetry(T* data_out, const Shape& shape, const Stride& strides,
                    std::vector<int64_t> dims, int64_t elem_count) {
  // const int NDIM = out_shape.size();
  const oneflow::NdIndexStrideOffsetHelper<int64_t, NDIM> helper(strides.data(), strides.size());
  std::sort(dims.begin(), dims.end());
  int64_t last_dim = dims.back();
  int64_t last_dim_size = shape[last_dim];
  int64_t last_dim_half = last_dim_size / 2;

  std::vector<int64_t> indices(shape.size());
  for (int offset = 0; offset < elem_count; offset++) {
    helper.OffsetToNdIndex(offset, indices.data(), indices.size());
    if (indices[last_dim] <= last_dim_half) { continue; }

    int64_t cur_last_dim_index = indices[last_dim];
    // get symmetric
    indices[last_dim] = last_dim_size - cur_last_dim_index;
    int64_t symmetric_offset = helper.NdIndexToOffset(indices.data(), indices.size());

    // conj
    data_out[offset] = std::conj(data_out[symmetric_offset]);
  }
}

template<typename T>
void conj_symmetry(T* data_out, const Shape& shape, const Stride& strides,
                   const std::vector<int64_t>& dims, int64_t elem_count) {
  void (*func)(T* /*data_out*/, const Shape& /*shape*/, const Stride& /*strides*/,
               const std::vector<int64_t>& /*dims*/, int64_t /*elem_count*/) = nullptr;

  switch (shape.size()) {
    case 1: _conj_symmetry<T, 1>; break;
    case 2: _conj_symmetry<T, 2>; break;
    case 3: _conj_symmetry<T, 3>; break;
    case 4: _conj_symmetry<T, 4>; break;
    case 5: _conj_symmetry<T, 5>; break;
    case 6: _conj_symmetry<T, 6>; break;
    case 7: _conj_symmetry<T, 7>; break;
    case 8: _conj_symmetry<T, 8>; break;
    case 9: _conj_symmetry<T, 9>; break;
    case 10: _conj_symmetry<T, 10>; break;
    case 11: _conj_symmetry<T, 11>; break;
    case 12: _conj_symmetry<T, 12>; break;
    default: UNIMPLEMENTED(); break;
  }
  _conj_symmetry(data_out, shape, strides, dims, elem_count);
}

template<DeviceType device_type, typename IN, typename OUT, typename dtype>
struct FftC2CKernelUtil {
  static void FftC2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, fft_norm_mode normalization);
};

template<DeviceType device_type, typename IN, typename OUT, typename dtype>
struct FftR2CKernelUtil {
  static void FftR2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, fft_norm_mode normalization);
};

#define INSTANTIATE_FFTC2C_KERNEL_UTIL(device_type, in_type_pair, out_type_pair, dtype) \
  template struct FftC2CKernelUtil<device_type, in_type_pair, out_type_pair, dtype>;

#define INSTANTIATE_FFTR2C_KERNEL_UTIL(device_type, in_type_pair, out_type_pair, dtype) \
  template struct FftR2CKernelUtil<device_type, in_type_pair, out_type_pair, dtype>;

}  // namespace oneflow
#endif  // ONEFLOW_USER_KERNEL_UTIL_H_