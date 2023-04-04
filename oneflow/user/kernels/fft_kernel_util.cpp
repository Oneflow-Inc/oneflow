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
#include "oneflow/core/common/preprocessor.h"
#include "pocketfftplan.h"

namespace oneflow {

template<typename T>
struct FftC2CKernelUtil<DeviceType::kCPU, T> {
  static void FftC2CForward(ep::Stream* stream, const std::complex<T>* data_in,
                            std::complex<T>* data_out, const Shape& input_shape,
                            const Shape& output_shape, const Stride& input_stride,
                            const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, fft_norm_mode normalization) {
    PocketFFtParams<T> params(input_shape, output_shape, input_stride, output_stride, dims, forward,
                              compute_fct<T>(input_shape, dims, normalization) /*1.f*/,
                              FFT_EXCUTETYPE::C2C);
    PocketFFtConfig<T> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename T>
struct FftR2CKernelUtil<DeviceType::kCPU, T> {
  static void FftR2CForward(ep::Stream* stream, const T* data_in, std::complex<T>* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, fft_norm_mode normalization) {
    // get temp buffer ? or use out, must be sure `out` is contiguos?

    // get last dim half size

    // do r2c, get half size fft out
    PocketFFtParams<T> params(input_shape, output_shape, input_stride, output_stride, dims, forward,
                              compute_fct<T>(input_shape, dims, normalization) /*1.f*/,
                              FFT_EXCUTETYPE::R2C);
    PocketFFtConfig<T> config(params);
    config.excute(data_in, data_out);

    // convert_to_doublesized
  }
};

template<typename T>
struct FftC2RKernelUtil<DeviceType::kCPU, T> {
  static void FftC2RForward(ep::Stream* stream, const std::complex<T>* data_in, T* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride,
                            int64_t last_dim_size, const std::vector<int64_t>& dims,
                            fft_norm_mode normalization) {
    PocketFFtParams<T> params(
        input_shape, output_shape, input_stride, output_stride, dims, /*is_forward=*/false,
        compute_fct<T>(output_shape, dims, normalization) /*1.f*/, FFT_EXCUTETYPE::C2R);
    PocketFFtConfig<T> config(params);
    config.excute(data_in, data_out);
  }
};

template struct FftC2CKernelUtil<DeviceType::kCPU, float>;
template struct FftC2CKernelUtil<DeviceType::kCPU, double>;

template struct FftR2CKernelUtil<DeviceType::kCPU, float>;
template struct FftR2CKernelUtil<DeviceType::kCPU, double>;

template struct FftC2RKernelUtil<DeviceType::kCPU, float>;
template struct FftC2RKernelUtil<DeviceType::kCPU, double>;

}  // namespace oneflow