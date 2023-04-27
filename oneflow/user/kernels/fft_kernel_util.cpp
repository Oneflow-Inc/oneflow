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
#include "oneflow/user/kernels/fft_kernel_util.h"
#include <type_traits>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "pocketfftplan.h"

namespace oneflow {

template<typename T, int NDIM>
static void _conj_symmetry_cpu(T* data_out, const Shape& shape, const std::vector<int64_t>& strides,
                           const int64_t last_dim, int64_t elem_count) {
  const oneflow::NdIndexStrideOffsetHelper<int64_t, NDIM> helper(strides.data(), NDIM);
  // NOTE: dims must be sorted
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
struct FillConjSymmetryUtil<DeviceType::kCPU, T>{
  static void FillConjSymmetryForward(ep::Stream* stream, T* data_out, const Shape& shape, const Stride& strides,
                                      const int64_t last_dim, int64_t elem_count){
    void (*func)(T* /*data_out*/, const Shape& /*shape*/, const std::vector<int64_t>& /*strides*/,
                const int64_t /*last_dim*/, int64_t /*elem_count*/) = nullptr;

    switch (shape.size()) {
      case 1: func = _conj_symmetry_cpu<T, 1>; break;
      case 2: func = _conj_symmetry_cpu<T, 2>; break;
      case 3: func = _conj_symmetry_cpu<T, 3>; break;
      case 4: func = _conj_symmetry_cpu<T, 4>; break;
      case 5: func = _conj_symmetry_cpu<T, 5>; break;
      case 6: func = _conj_symmetry_cpu<T, 6>; break;
      case 7: func = _conj_symmetry_cpu<T, 7>; break;
      case 8: func = _conj_symmetry_cpu<T, 8>; break;
      case 9: func = _conj_symmetry_cpu<T, 9>; break;
      case 10: func = _conj_symmetry_cpu<T, 10>; break;
      case 11: func = _conj_symmetry_cpu<T, 11>; break;
      case 12: func = _conj_symmetry_cpu<T, 12>; break;
      default: UNIMPLEMENTED(); break;
    }
    std::vector<int64_t> strides_vec(strides.begin(), strides.end());
    func(data_out, shape, strides_vec, last_dim, elem_count);
  }
};

template<typename T, typename FCT_TYPE>
struct FftC2CKernelUtil<DeviceType::kCPU, T, FCT_TYPE> {
  static void FftC2CForward(ep::Stream* stream, 
                            const T* data_in, T* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride,
                            bool forward, const std::vector<int64_t>& dims, FCT_TYPE norm_fct, DataType real_type) {
    PocketFFtParams<FCT_TYPE> params(
        input_shape, output_shape, input_stride, output_stride, dims, forward,
        norm_fct /*1.f*/, FFT_EXCUTETYPE::C2C);
    PocketFFtConfig<FCT_TYPE> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename IN, typename OUT>
struct FftR2CKernelUtil<DeviceType::kCPU, IN, OUT> {
  static void FftR2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride,
                            bool forward, const std::vector<int64_t>& dims, IN norm_fct,
                            DataType real_type) {
    PocketFFtParams<IN> params(input_shape, output_shape, input_stride, output_stride, dims, forward,
                              norm_fct /*1.f*/, FFT_EXCUTETYPE::R2C);
    PocketFFtConfig<IN> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename IN, typename OUT>
struct FftC2RKernelUtil<DeviceType::kCPU, IN, OUT> {
  static void FftC2RForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            int64_t last_dim_size, const std::vector<int64_t>& dims,
                            OUT norm_fct, DataType real_type) {
    PocketFFtParams<OUT> params(
        input_shape, output_shape, input_stride, output_stride, dims, /*is_forward=*/false,
        norm_fct /*1.f*/, FFT_EXCUTETYPE::C2R);
    PocketFFtConfig<OUT> config(params);
    config.excute(data_in, data_out);
  }
};

template struct FillConjSymmetryUtil<DeviceType::kCPU, std::complex<float>>;
template struct FillConjSymmetryUtil<DeviceType::kCPU, std::complex<double>>;


template struct FftC2CKernelUtil<DeviceType::kCPU, std::complex<float>, float>;
template struct FftC2CKernelUtil<DeviceType::kCPU, std::complex<double>, double>;

template struct FftR2CKernelUtil<DeviceType::kCPU, float, std::complex<float>>;
template struct FftR2CKernelUtil<DeviceType::kCPU, double, std::complex<double>>;

template struct FftC2RKernelUtil<DeviceType::kCPU, std::complex<float>, float>;
template struct FftC2RKernelUtil<DeviceType::kCPU, std::complex<double>, double>;

}  // namespace oneflow