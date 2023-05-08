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

#include "pocketfft_hdronly.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace {

enum class FFT_EXCUTETYPE { R2C, C2C, C2R };

template<typename dtype>
struct PocketFFtParams {
  bool IsForward;
  FFT_EXCUTETYPE excute_type;
  dtype fct;
  pocketfft::shape_t axes;
  pocketfft::stride_t in_stridef;
  pocketfft::stride_t out_stridef;
  pocketfft::shape_t input_shape;
  pocketfft::shape_t output_shape;
  PocketFFtParams() = default;
  PocketFFtParams(const Shape& in_shape, const Shape& out_shape, const Stride& in_stride,
                  const Stride& out_stride, const std::vector<int64_t>& dims, const bool is_forward,
                  const dtype f, FFT_EXCUTETYPE type)
      : IsForward(is_forward),
        excute_type(type),
        fct(f),
        axes(dims.begin(), dims.end()),
        in_stridef(in_stride.begin(), in_stride.end()),
        out_stridef(out_stride.begin(), out_stride.end()) {
    input_shape.resize(in_shape.size());
    output_shape.resize(out_shape.size());

    std::copy(in_shape.begin(), in_shape.end(), input_shape.begin());
    std::copy(out_shape.begin(), out_shape.end(), output_shape.begin());

    // calc element size
    size_t in_elemsize = type == FFT_EXCUTETYPE::C2C || type == FFT_EXCUTETYPE::C2R
                             ? sizeof(std::complex<dtype>)
                             : sizeof(dtype);
    size_t out_elemsize = type == FFT_EXCUTETYPE::R2C || type == FFT_EXCUTETYPE::C2C
                              ? sizeof(std::complex<dtype>)
                              : sizeof(dtype);
    for (auto& s : in_stridef) { s *= in_elemsize; }
    for (auto& s : out_stridef) { s *= out_elemsize; }
  }
};

template<typename dtype>
class PocketFFtConfig {
 public:
  PocketFFtConfig(const PocketFFtConfig&) = delete;
  PocketFFtConfig& operator=(PocketFFtConfig const&) = delete;

  explicit PocketFFtConfig(const PocketFFtParams<dtype>& params) : fftparams(params) {}

  void excute(const std::complex<dtype>* in, std::complex<dtype>* out) {
    pocketfft::c2c(fftparams.input_shape, fftparams.in_stridef, fftparams.out_stridef,
                   fftparams.axes, fftparams.IsForward, in, out, fftparams.fct);
  }

  void excute(const dtype* in, std::complex<dtype>* out) {
    pocketfft::r2c(fftparams.input_shape, fftparams.in_stridef, fftparams.out_stridef,
                   fftparams.axes, fftparams.IsForward, in, out, fftparams.fct);
  }

  void excute(const std::complex<dtype>* in, dtype* out) {
    pocketfft::c2r(fftparams.output_shape, fftparams.in_stridef, fftparams.out_stridef,
                   fftparams.axes, fftparams.IsForward, in, out, fftparams.fct);
  }

 private:
  PocketFFtParams<dtype> fftparams;
};

}  // namespace

}  // namespace oneflow