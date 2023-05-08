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
#include "pocketfftplan.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {

template<typename T>
static void _conj_symmetry_cpu(T* data_out, const Shape& shape, const std::vector<int64_t>& strides,
                               const int64_t last_dim, int64_t elem_count) {
  const oneflow::NdIndexStrideOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> helper(strides.data(),
                                                                                shape.size());
  // NOTE: dims must be sorted
  int64_t last_dim_size = shape[last_dim];
  int64_t last_dim_half = last_dim_size / 2;

  int64_t ndim = shape.size();
  std::vector<int64_t> indices(ndim);
  for (int offset = 0; offset < elem_count; offset++) {
    helper.OffsetToNdIndex(offset, indices.data(), ndim);
    if (indices[last_dim] <= last_dim_half) { continue; }

    int64_t cur_last_dim_index = indices[last_dim];
    // get symmetric
    indices[last_dim] = last_dim_size - cur_last_dim_index;
    int64_t symmetric_offset = helper.NdIndexToOffset(indices.data(), ndim);

    // conj
    data_out[offset] = std::conj(data_out[symmetric_offset]);
  }
}

template<typename T>
struct FillConjSymmetryUtil<DeviceType::kCPU, T> {
  static void FillConjSymmetryForward(ep::Stream* stream, T* data_out, const Shape& shape,
                                      const Stride& strides, const int64_t last_dim,
                                      int64_t elem_count) {
    std::vector<int64_t> strides_vec(strides.begin(), strides.end());
    _conj_symmetry_cpu(/*data_out*/ data_out, /*shape*/ shape, /*strides*/ strides_vec,
                       /*last_dim*/ last_dim, /*elem_count*/ elem_count);
  }
};

template<typename real_type, typename complex_type>
struct ComplexConvertUtil<DeviceType::kCPU, real_type, complex_type> {
  static void ConvertToDoubleSized(ep::Stream* stream, const complex_type* in, complex_type* dst,
                                   size_t len, size_t n) {
    size_t fact_len = 2 * len - 2;  // input_shape.back()
    for (int i = 0; i < n; i++) {
      int index_x = i / fact_len;
      int index_y = i % fact_len;
      if (index_y == 0) {
        dst[i] = in[index_x * len];
      } else if (index_y == len - 1) {
        dst[i] = in[(index_x + 1) * len - 1];
      } else if (index_y < len - 1 && index_y > 0) {
        dst[i] = in[index_x * len + index_y];
      } else {
        auto index = (index_x + 2) * len - index_y - 2;
        auto realvalue = in[index].real();
        dst[i].real(realvalue);
        auto imagvalue = -in[index].imag();
        dst[i].imag(imagvalue);
      }
    }
  }
  static void ConvertComplexToReal(ep::Stream* stream, const complex_type* in, real_type* out,
                                   size_t n) {
    for (int i = 0; i < n; i++) {
      out[2 * i] = in[i].real();
      out[2 * i + 1] = in[i].imag();
    }
  }
};

template<typename T, typename FCT_TYPE>
struct FftC2CKernelUtil<DeviceType::kCPU, T, FCT_TYPE> {
  static void FftC2CForward(ep::Stream* stream, const T* data_in, T* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, FCT_TYPE norm_fct,
                            DataType real_type) {
    PocketFFtParams<FCT_TYPE> params(input_shape, output_shape, input_stride, output_stride, dims,
                                     forward, norm_fct /*1.f*/, FFT_EXCUTETYPE::C2C);
    PocketFFtConfig<FCT_TYPE> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename IN, typename OUT>
struct FftR2CKernelUtil<DeviceType::kCPU, IN, OUT> {
  static void FftR2CForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            const std::vector<int64_t>& dims, IN norm_fct, DataType real_type) {
    PocketFFtParams<IN> params(input_shape, output_shape, input_stride, output_stride, dims,
                               forward, norm_fct /*1.f*/, FFT_EXCUTETYPE::R2C);
    PocketFFtConfig<IN> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename IN, typename OUT>
struct FftC2RKernelUtil<DeviceType::kCPU, IN, OUT> {
  static void FftC2RForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                            const Shape& input_shape, const Shape& output_shape,
                            const Stride& input_stride, const Stride& output_stride, bool forward,
                            int64_t last_dim_size, const std::vector<int64_t>& dims, OUT norm_fct,
                            DataType real_type) {
    PocketFFtParams<OUT> params(input_shape, output_shape, input_stride, output_stride, dims,
                                /*is_forward=*/false, norm_fct /*1.f*/, FFT_EXCUTETYPE::C2R);
    PocketFFtConfig<OUT> config(params);
    config.excute(data_in, data_out);
  }
};

template<typename IN, typename OUT>
struct FftStftKernelUtil<DeviceType::kCPU, IN, OUT> {
  static void FftStftForward(ep::Stream* stream, const IN* data_in, OUT* data_out,
                             const Shape& input_shape, const Shape& output_shape,
                             const Stride& input_stride, const Stride& output_stride, bool forward,
                             const std::vector<int64_t>& axes, IN norm_fct, int64_t len,
                             int64_t dims, int64_t batch) {
    PocketFFtParams<IN> params(input_shape, output_shape, input_stride, output_stride, axes,
                               forward, norm_fct /*1.f*/, FFT_EXCUTETYPE::R2C);
    PocketFFtConfig<IN> config(params);
    int64_t in_offset = len;
    int64_t out_offset = len / 2 + 1;
    for (int j = 0; j < dims; j++) {
      for (int i = 0; i < batch; i++) {
        const IN* in = data_in + j * batch * in_offset + i * in_offset;
        OUT* out = data_out + j * batch * out_offset + i * out_offset;
        config.excute(in, out);
      }
    }
  }
};
template struct FillConjSymmetryUtil<DeviceType::kCPU, std::complex<float>>;
template struct FillConjSymmetryUtil<DeviceType::kCPU, std::complex<double>>;

template struct ComplexConvertUtil<DeviceType::kCPU, float, std::complex<float>>;
template struct ComplexConvertUtil<DeviceType::kCPU, double, std::complex<double>>;

template struct FftC2CKernelUtil<DeviceType::kCPU, std::complex<float>, float>;
template struct FftC2CKernelUtil<DeviceType::kCPU, std::complex<double>, double>;

template struct FftR2CKernelUtil<DeviceType::kCPU, float, std::complex<float>>;
template struct FftR2CKernelUtil<DeviceType::kCPU, double, std::complex<double>>;

template struct FftC2RKernelUtil<DeviceType::kCPU, std::complex<float>, float>;
template struct FftC2RKernelUtil<DeviceType::kCPU, std::complex<double>, double>;

template struct FftStftKernelUtil<DeviceType::kCPU, float, std::complex<float>>;
template struct FftStftKernelUtil<DeviceType::kCPU, double, std::complex<double>>;
}  // namespace oneflow