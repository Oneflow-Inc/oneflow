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

template<DeviceType device_type, typename T, typename FCT_TYPE>
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
    double norm_fct = ctx->Attr<double>("norm_fct");
    
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");

    const T* input_ptr = input->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());

    if (input->data_type() == kComplex64){
      FftC2CKernelUtil<device_type, T, FCT_TYPE>::FftC2CForward(ctx->stream(), input_ptr, out_ptr,
                                                      input_shape, out_shape, input->stride(),
                                                      out->stride(), forward, dims, static_cast<FCT_TYPE>(norm_fct),
                                                      DataType::kFloat);      
    }
    else if(input->data_type() == kComplex128){
      FftC2CKernelUtil<device_type, T, FCT_TYPE>::FftC2CForward(ctx->stream(), input_ptr, out_ptr,
                                                      input_shape, out_shape, input->stride(),
                                                      out->stride(), forward, dims, static_cast<FCT_TYPE>(norm_fct),
                                                      DataType::kDouble);      
    }
    else {
      Error::RuntimeError() << "expects kComplex64 or kComplex128, but got " << input->data_type();
    }
  }
};


template<DeviceType device_type, typename dtype_in, typename dtype_out>
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
    double norm_fct = ctx->Attr<double>("norm_fct");
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");
    const dtype_in* input_ptr = input->dptr<dtype_in>();
    dtype_out* out_ptr = out->mut_dptr<dtype_out>();

    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());



    if (input->data_type() == kFloat || input->data_type() == kDouble) {
      FftR2CKernelUtil<device_type, dtype_in, dtype_out>::FftR2CForward(
          ctx->stream(), input_ptr, out_ptr, 
          input_shape, out_shape, input->stride(), out->stride(), 
          /*forward=*/true, dims, norm_fct);
    } else {
      Error::RuntimeError() << "expects kFloat or kDouble, but gets " << input->data_type();
    }

    // if (!onesided) { conj_symmetry(out_ptr, out_shape, out->stride(), dims, out_shape.elem_cnt()); }
    if (!onesided){
      FillConjSymmetryUtil<device_type, dtype_out>::FillConjSymmetryForward(
        ctx->stream(), out_ptr, out_shape, out->stride(), dims.back(), out_shape.elem_cnt());
    }
  }
};

#if 0
template<typename dtype_in, typename dtype_out>
class FftR2CCudaKernel final : public user_op::OpKernel {
 public:
  FftR2CCudaKernel() = default;
  ~FftR2CCudaKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::cout << "=========== [FftR2CCudaKernel] in ==================" << std::endl;

    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool forward = ctx->Attr<bool>("forward");
    bool onesided = ctx->Attr<bool>("onesided");
    const std::string& norm_str = ctx->Attr<std::string>("norm");
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");
    const dtype_in* input_ptr = input->dptr<dtype_in>();
    dtype_out* out_ptr = out->mut_dptr<dtype_out>();
    // TO-DO:
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    // =================


    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());
    fft_norm_mode norm_mode = norm_from_string(norm_str, forward);

    // get last dim half size
    if (onesided) {
      int64_t last_dim = dims.back();
      int64_t last_dim_halfsize = (input_shape[last_dim]) / 2 + 1;
      out_shape[last_dim] = last_dim_halfsize;
    }

    if (input->data_type() == kFloat || input->data_type() == kDouble) {
      FftR2CKernelUtil<DeviceType::kCPU, dtype_in, dtype_out>::FftR2CForward(
          ctx->stream(), input_ptr, out_ptr, input_shape, out_shape, input->stride(), out->stride(),
          /*forward=*/true, dims, norm_mode);
    } else {
      Error::RuntimeError() << "expects kFloat or kDouble, but gets " << input->data_type();
    }

    if (!onesided) { conj_symmetry(out_ptr, out_shape, out->stride(), dims, out_shape.elem_cnt()); }
  }
};
#endif

template<DeviceType device_type, typename dtype_in, typename dtype_out>
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
    double norm_fct = ctx->Attr<double>("norm_fct");
    const std::vector<int64_t>& dims = ctx->Attr<std::vector<int64_t>>("dims");

    const dtype_in* input_ptr = input->dptr<dtype_in>();
    dtype_out* out_ptr = out->mut_dptr<dtype_out>();

    Shape input_shape(input->shape_view());
    Shape out_shape(out->shape_view());

    out_shape[dims.back()] = last_dim_size;

    if (input->data_type() == kComplex64 || input->data_type() == kComplex128) {
      FftC2RKernelUtil<device_type, dtype_in, dtype_out>::FftC2RForward(
          ctx->stream(), input_ptr, out_ptr, input_shape, out_shape, 
          input->stride(), out->stride(),
          /*last_dim_size=*/last_dim_size, dims, norm_fct);
    } else {
      Error::RuntimeError() << "expects kComplex64 or kComplex128, but gets " << input->data_type();
    }
  }
};


template<DeviceType device_type, typename dtype_in, typename dtype_out>
class StftCpuKernel final : public user_op::OpKernel {
 public:
  StftCpuKernel() = default;
  ~StftCpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_STFT_CPU_KERNEL(dtype_in, dtype_out)                                         \
  REGISTER_USER_KERNEL("stft")                                                                \
      .SetCreateFn<StftCpuKernel<DeviceType::kCPU, dtype_in, dtype_out>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == kCPU)                                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype_in>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const Shape& output_shape = ctx->InputShape("output", 0);                             \
        const bool return_complex = ctx->Attr<bool>("return_complex");                        \
        const bool onesided = ctx->Attr<bool>("onesided");                                    \
        int64_t output_elem_cnt =                                                             \
            return_complex ? output_shape.elem_cnt() : output_shape.elem_cnt() / 2;           \
        const int64_t output_bytes = (output_elem_cnt * sizeof(dtype_out));      \
        return onesided ? output_bytes : 2 * output_bytes;                                    \
      });

REGISTER_STFT_CPU_KERNEL(double, std::complex<double>)
REGISTER_STFT_CPU_KERNEL(float, std::complex<float>)
#ifdef WITH_CUDA
// TO-DO
// REGISTER_STFT_CUDA_KERNEL(...)
#endif

#define REGISTER_FFTC2C_KERNELS(device_type, dtype, fct_type)                                                \
  REGISTER_USER_KERNEL("fft_c2c").SetCreateFn<FftC2CKernel<device_type, dtype, fct_type>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device_type)                                                    \
      && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)                      \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))

REGISTER_FFTC2C_KERNELS(DeviceType::kCPU, std::complex<float>, float);
REGISTER_FFTC2C_KERNELS(DeviceType::kCPU, std::complex<double>, double);
#ifdef WITH_CUDA
REGISTER_FFTC2C_KERNELS(DeviceType::kCUDA, cuComplex, float);
REGISTER_FFTC2C_KERNELS(DeviceType::kCUDA, cuDoubleComplex, double);
#endif

#define REGISTER_FFTR2C_KERNELS(device_type, dtype_in, dtype_out)                                 \
  REGISTER_USER_KERNEL("fft_r2c")                                                            \
      .SetCreateFn<FftR2CKernel<device_type, dtype_in, dtype_out>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type)                                  \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype_in>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype_out>::value))

REGISTER_FFTR2C_KERNELS(DeviceType::kCPU, float, std::complex<float>);
REGISTER_FFTR2C_KERNELS(DeviceType::kCPU, double, std::complex<double>);
#ifdef WITH_CUDA
#endif

#define REGISTER_FFTC2R_KERNELS(device_type, dtype_in, dtype_out)                                 \
  REGISTER_USER_KERNEL("fft_c2r")                                                            \
      .SetCreateFn<FftC2RKernel<device_type, dtype_in, dtype_out>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type)                                  \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype_in>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype_out>::value))

REGISTER_FFTC2R_KERNELS(DeviceType::kCPU, std::complex<float>, float);
REGISTER_FFTC2R_KERNELS(DeviceType::kCPU, std::complex<double>, double);
#ifdef WITH_CUDA
#endif
}  // namespace oneflow