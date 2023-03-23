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

template<typename IN, typename OUT, typename dtype>
struct FftC2CKernelUtil<DeviceType::kCPU, IN, OUT, dtype> {
  static void FftC2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, fft_norm_mode normalization) {
    PocketFFtParams<dtype> params(
        input_shape, output_shape, input_stride, output_stride, dims, forward,
        compute_fct<dtype>(input_shape, dims, normalization) /*1.f*/, FFT_EXCUTETYPE::C2C);
    PocketFFtConfig<dtype> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename IN, typename OUT, typename dtype>
struct FftR2CKernelUtil<DeviceType::kCPU, IN, OUT, dtype> {
  static void FftR2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, fft_norm_mode normalization) {
    // get temp buffer ? or use out, must be sure `out` is contiguos?

    // get last dim half size

    // do r2c, get half size fft out
    PocketFFtParams<dtype> params(
        input_shape, output_shape, input_stride, output_stride, dims, forward,
        compute_fct<dtype>(input_shape, dims, normalization) /*1.f*/, FFT_EXCUTETYPE::R2C);
    PocketFFtConfig<dtype> config(params);
    config.excute(data_in, data_out);

    // convert_to_doublesized
  }
};


template struct FftC2CKernelUtil<DeviceType::kCPU, std::complex<float>, std::complex<float>, float>;
template struct FftC2CKernelUtil<DeviceType::kCPU, std::complex<double>, std::complex<double>, double>;

template struct FftR2CKernelUtil<DeviceType::kCPU, float, std::complex<float>, float>;
template struct FftR2CKernelUtil<DeviceType::kCPU, double, std::complex<double>, double>; 

}  // namespace oneflow