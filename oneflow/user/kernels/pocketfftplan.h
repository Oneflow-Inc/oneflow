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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "pocketfft_hdronly.h"
#include "oneflow/core/kernel/kernel.h"
using namespace pocketfft;

namespace oneflow {
namespace {

enum class FFT_EXCUTETYPE { R2C, C2C };

template<typename IN, typename OUT>
struct PocketFFtParams {
  shape_t input_shape;
  shape_t output_shape;
  stride_t in_stridef;
  stride_t out_stridef;
  shape_t axes;
  bool IsForward;
  FFT_EXCUTETYPE excute_type;
  IN fct;
  PocketFFtParams() = default;
  PocketFFtParams(const Shape& in_shape, const Shape& out_shape, const bool is_froward, const IN f,
                  FFT_EXCUTETYPE type)
      : IsForward(is_froward), excute_type(type), fct(f) {
    input_shape.resize(in_shape.size());
    output_shape.resize(out_shape.size());
    in_stridef.resize(input_shape.size());
    out_stridef.resize(output_shape.size());
    axes.resize(input_shape.size());

    std::copy(in_shape.begin(), in_shape.end(), input_shape.begin());
    std::copy(out_shape.begin(), out_shape.end(), output_shape.begin());
    std::iota(axes.begin(), axes.end(), 0);

    size_t out_tmpf = sizeof(OUT);
    size_t in_tmpf = sizeof(IN);
    for (int i = input_shape.size() - 1; i >= 0; --i) {
      in_stridef[i] = in_tmpf;
      in_tmpf *= input_shape[i];
      out_stridef[i] = out_tmpf;
      out_tmpf *= output_shape[i];
    }
  }
};

template<typename IN, typename OUT>
class PocketFFtConfig {
 public:
  PocketFFtConfig(const PocketFFtConfig&) = delete;
  PocketFFtConfig& operator=(PocketFFtConfig const&) = delete;

  explicit PocketFFtConfig(const PocketFFtParams<IN, OUT>& params) : fftparams(params) {}

  void excute(const IN* in, OUT* out, int64_t dims, int64_t batch, int64_t len) {
    int64_t in_offset = len;
    int64_t out_offset = len / 2 + 1;
    for (int j = 0; j < dims; j++) {
      for (int i = 0; i < batch; i++) {
        const IN* data_in = in + j * batch * in_offset + i * in_offset;
        OUT* data_out = out + j * batch * out_offset + i * out_offset;
        switch (fftparams.excute_type) {
          case FFT_EXCUTETYPE::R2C:
            r2c(fftparams.input_shape, fftparams.in_stridef, fftparams.out_stridef, fftparams.axes,
                fftparams.IsForward, data_in, data_out, fftparams.fct);
            break;

          case FFT_EXCUTETYPE::C2C:
            // c2c(fftparams.input_shape, fftparams.in_stridef, fftparams.out_stridef,
            // fftparams.axes, fftparams.IsForward, in,
            //     out, fftparams.fct);
            break;
          default: break;
        }
      }
    }
  }

 private:
  PocketFFtParams<IN, OUT> fftparams;
};

}  // namespace

}  // namespace oneflow