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
#include <complex>
#include <cstdint>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/user/kernels/fft_kernel_util.h"
#include "pocketfftplan.h"
using namespace pocketfft;
namespace oneflow {

namespace {

// len = input_shape.back() / 2 + 1
// n = output_shape.elem_cnt() / 2
template<typename T>
void convert_to_doublesized(const std::complex<T>* in, std::complex<T>* dst, size_t len, size_t n) {
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

template<typename T>
void comvert_to_real(const std::complex<T>* in, T* out, size_t n) {
  for (int i = 0; i < n; i++) {
    out[2 * i] = in[i].real();
    out[2 * i + 1] = in[i].imag();
  }
}

}  // namespace

template<DeviceType device_type, typename T>
class FftC2CKernel final : public user_op::OpKernel {
 public:
  FftC2CKernel() = default;
  ~FftC2CKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::cout << "=========== [FftC2CKernel] in ==================" << std::endl;

    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool forward = ctx->Attr<bool>("forward");
    bool is_grad_fn = ctx->Attr<bool>("is_grad_fn");
    const std::string& norm_str = ctx->Attr<std::string>("norm");
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");

    const std::complex<T>* input_ptr = input->dptr<std::complex<T>>();
    std::complex<T>* out_ptr = out->mut_dptr<std::complex<T>>();

    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());

    fft_norm_mode norm_mode = fft_norm_mode::none;
    if (!is_grad_fn) {
      norm_mode = norm_from_string(norm_str, forward);
    } else {
      norm_mode = norm_from_string(norm_str, !forward);
    }

    if (input->data_type() == kComplex64) {
      FftC2CKernelUtil<device_type, T>::FftC2CForward(ctx->stream(), input_ptr, out_ptr,
                                                      input_shape, out_shape, input->stride(),
                                                      out->stride(), forward, dims, norm_mode);
    } else if (input->data_type() == kComplex128) {
      FftC2CKernelUtil<device_type, T>::FftC2CForward(ctx->stream(), input_ptr, out_ptr,
                                                      input_shape, out_shape, input->stride(),
                                                      out->stride(), forward, dims, norm_mode);
    } else {
      Error::RuntimeError() << "expects kComplex64 or kComplex128, but got " << input->data_type();
    }
  }
};

template<DeviceType device_type, typename T>
class FftR2CKernel final : public user_op::OpKernel {
 public:
  FftR2CKernel() = default;
  ~FftR2CKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::cout << "=========== [FftR2CKernel] in ==================" << std::endl;

    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool forward = ctx->Attr<bool>("forward");
    bool onesided = ctx->Attr<bool>("onesided");
    const std::string& norm_str = ctx->Attr<std::string>("norm");
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");
    const T* input_ptr = input->dptr<T>();
    std::complex<T>* out_ptr = out->mut_dptr<std::complex<T>>();

    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());
    fft_norm_mode norm_mode = norm_from_string(norm_str, forward);
    std::cout << "=========== [FftR2CKernel] 1 ==================" << std::endl;

    // get last dim half size
    if (onesided) {
      int64_t last_dim = dims.back();
      int64_t last_dim_halfsize = (input_shape[last_dim]) / 2 + 1;
      out_shape[last_dim] = last_dim_halfsize;
    }
    std::cout << "=========== [FftR2CKernel] 2 ==================" << std::endl;

    if (input->data_type() == kFloat) {
      FftR2CKernelUtil<device_type, T>::FftR2CForward(
          ctx->stream(), input_ptr, out_ptr, input_shape, out_shape, input->stride(), out->stride(),
          /*forward=*/true, dims, norm_mode);
    } else if (input->data_type() == kDouble) {
      FftR2CKernelUtil<device_type, T>::FftR2CForward(
          ctx->stream(), input_ptr, out_ptr, input_shape, out_shape, input->stride(), out->stride(),
          /*forward=*/true, dims, norm_mode);
    } else {
      Error::RuntimeError() << "expects kFloat or kDouble, but gets " << input->data_type();
    }
    std::cout << "=========== [FftR2CKernel] 3 ==================" << std::endl;

    if (!onesided) { conj_symmetry(out_ptr, out_shape, out->stride(), dims, out_shape.elem_cnt()); }
    std::cout << "=========== [FftR2CKernel] 4 ==================" << std::endl;
  }
};

template<DeviceType device_type, typename T>
class FftC2RKernel final : public user_op::OpKernel {
 public:
  FftC2RKernel() = default;
  ~FftC2RKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::cout << "=========== [FftC2RKernel] in ==================" << std::endl;

    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t last_dim_size = ctx->Attr<int64_t>("last_dim_size");
    bool forward = ctx->Attr<bool>("forward");
    const std::string& norm_str = ctx->Attr<std::string>("norm");
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");
    std::cout << "=========== [FftC2RKernel] get attr ==================" << std::endl;

    const std::complex<T>* input_ptr = input->dptr<std::complex<T>>();
    T* out_ptr = out->mut_dptr<T>();

    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());
    fft_norm_mode norm_mode = norm_from_string(norm_str, forward);
    std::cout << "=========== [FftC2RKernel] get norm ==================" << std::endl;

    out_shape[dims.back()] = last_dim_size;

    if (input->data_type() == kComplex64 || input->data_type() == kComplex128) {
      FftC2RKernelUtil<device_type, T>::FftC2RForward(
          ctx->stream(), input_ptr, out_ptr, input_shape, out_shape, input->stride(), out->stride(),
          /*last_dim_size=*/last_dim_size, dims, norm_mode);
    } else {
      Error::RuntimeError() << "expects kComplex64 or kComplex128, but gets " << input->data_type();
    }
  }
};

#if 1
// template<typename IN, typename OUT>
template<DeviceType device_type, typename T>
class StftCpuKernel final : public user_op::OpKernel {
 public:
  StftCpuKernel() = default;
  ~StftCpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto normalized = ctx->Attr<bool>("normalized");
    const auto return_complex = ctx->Attr<bool>("return_complex");
    const bool onesized = ctx->Attr<bool>("onesided");

    const ShapeView& input_shape = input->shape_view();
    const ShapeView& output_shape = output->shape_view();
    const auto output_elem_cnt = output_shape.elem_cnt() / 2;

    int64_t dims = input_shape.At(0);
    int64_t batch = input_shape.At(1);
    int64_t len = input_shape.back();
    // const IN* data_in = input->dptr<IN>();
    const T* data_in = input->dptr<T>();
    T* data_out = output->mut_dptr<T>();

    // ===============
    // auto normalization = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
    // PocketFFtParams<IN, OUT> params(/*in_shape=*/Shape{len}, /*out_shape=*/Shape{len}, /*is_forward=*/true,
    //                                 /*f=*/compute_fct<IN>(len, normalization) /*1.f*/,
    //                                 /*type=*/FFT_EXCUTETYPE::R2C);
    // PocketFFtConfig<IN, OUT> config(params);
    // OUT* out_tmp_buffer = reinterpret_cast<OUT*>(tmp_buffer->mut_dptr<char>());
    // config.excute(data_in, out_tmp_buffer, dims, batch, len);
    // ===============
    auto normalization = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
    std::complex<T>* out_tmp_buffer = reinterpret_cast<std::complex<T>*>(tmp_buffer->mut_dptr<char>());
    Shape out_tmp_shape = Shape{len};
    Stride out_tmp_stride = Stride(out_tmp_shape);
    std::vector<int64_t> axes (out_tmp_shape.size());
    std::iota(axes.begin(), axes.end(), 0);
    FftStftKernelUtil<device_type, T>::FftStftForward(ctx->stream(), data_in, out_tmp_buffer, out_tmp_shape, 
                                                  out_tmp_shape, out_tmp_stride, out_tmp_stride, 
                                                  true, /*axes=*/axes, /*normalization=*/normalization,
                                                  /*len=*/len, /*dims=*/dims, /*batch=*/batch);

    if (!onesized) {
      std::complex<T>* doublesided_tmp_buffer =
          reinterpret_cast<std::complex<T>*>(tmp_buffer->mut_dptr<char>()) + output_elem_cnt;
      size_t last_dim_length = len / 2 + 1;
      size_t elem_conut = output_elem_cnt;
      convert_to_doublesized<T>(out_tmp_buffer, doublesided_tmp_buffer, last_dim_length,
                                 elem_conut);
      out_tmp_buffer = doublesided_tmp_buffer;
    }

    if (!return_complex) { comvert_to_real<T>(out_tmp_buffer, data_out, output_elem_cnt); }

    // if (!onesized) {
    //   OUT* doublesided_tmp_buffer =
    //       reinterpret_cast<OUT*>(tmp_buffer->mut_dptr<char>()) + output_elem_cnt;
    //   size_t last_dim_length = len / 2 + 1;
    //   size_t elem_conut = output_elem_cnt;
    //   convert_to_doublesized<IN>(out_tmp_buffer, doublesided_tmp_buffer, last_dim_length,
    //                              elem_conut);
    //   out_tmp_buffer = doublesided_tmp_buffer;
    // }

    // if (!return_complex) { comvert_to_real<IN>(out_tmp_buffer, data_out, output_elem_cnt); }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_STFT_CPU_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("stft")                                                              \
      .SetCreateFn<StftCpuKernel<DeviceType::kCPU, dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == kCPU)                                   \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const Shape& output_shape = ctx->InputShape("output", 0);                           \
        const bool return_complex = ctx->Attr<bool>("return_complex");                      \
        const bool onesided = ctx->Attr<bool>("onesided");                                  \
        int64_t output_elem_cnt =                                                           \
            return_complex ? output_shape.elem_cnt() : output_shape.elem_cnt() / 2;         \
        const int64_t output_bytes = (output_elem_cnt * sizeof(std::complex<dtype>));                   \
        return onesided ? output_bytes : 2 * output_bytes;                                  \
      });
// #define REGISTER_STFT_CPU_KERNEL(intype, outtype)                                           \
//   REGISTER_USER_KERNEL("stft")                                                              \
//       .SetCreateFn<StftCpuKernel<intype, outtype>>()                                        \
//       .SetIsMatchedHob((user_op::HobDeviceType() == kCPU)                                   \
//                        && (user_op::HobDataType("input", 0) == GetDataType<intype>::value)) \
//       .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
//         const Shape& output_shape = ctx->InputShape("output", 0);                           \
//         const bool return_complex = ctx->Attr<bool>("return_complex");                      \
//         const bool onesided = ctx->Attr<bool>("onesided");                                  \
//         int64_t output_elem_cnt =                                                           \
//             return_complex ? output_shape.elem_cnt() : output_shape.elem_cnt() / 2;         \
//         const int64_t output_bytes = (output_elem_cnt * sizeof(outtype));                   \
//         return onesided ? output_bytes : 2 * output_bytes;                                  \
//       });
REGISTER_STFT_CPU_KERNEL(double)
REGISTER_STFT_CPU_KERNEL(float)
// REGISTER_STFT_CPU_KERNEL(double, std::complex<double>)
// REGISTER_STFT_CPU_KERNEL(float, std::complex<float>)

#endif

#define REGISTER_FFTC2C_KERNELS(device, dtype)                                                \
  REGISTER_USER_KERNEL("fft_c2c").SetCreateFn<FftC2CKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                    \
      && (user_op::HobDataType("input", 0) == GetDataType<std::complex<dtype>>::value)        \
      && (user_op::HobDataType("out", 0) == GetDataType<std::complex<dtype>>::value))

REGISTER_FFTC2C_KERNELS(DeviceType::kCPU, float);
REGISTER_FFTC2C_KERNELS(DeviceType::kCPU, double);

#define REGISTER_FFTR2C_KERNELS(device, dtype)                                                \
  REGISTER_USER_KERNEL("fft_r2c").SetCreateFn<FftR2CKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                    \
      && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)                      \
      && (user_op::HobDataType("out", 0) == GetDataType<std::complex<dtype>>::value))

REGISTER_FFTR2C_KERNELS(DeviceType::kCPU, float);
REGISTER_FFTR2C_KERNELS(DeviceType::kCPU, double);

#define REGISTER_FFTC2R_KERNELS(device, dtype)                                                \
  REGISTER_USER_KERNEL("fft_c2r").SetCreateFn<FftC2RKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                    \
      && (user_op::HobDataType("input", 0) == GetDataType<std::complex<dtype>>::value)        \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))

REGISTER_FFTC2R_KERNELS(DeviceType::kCPU, float);
REGISTER_FFTC2R_KERNELS(DeviceType::kCPU, double);

}  // namespace oneflow