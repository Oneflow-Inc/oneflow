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
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

std::tuple<int32_t, int32_t, int32_t> CalcMNK(const ShapeView& a_shape, const ShapeView& out_shape,
                                              CBLAS_TRANSPOSE transpose_a) {
  int32_t num_axes = a_shape.NumAxes();
  int m = out_shape.At(num_axes - 2);
  int n = out_shape.At(num_axes - 1);
  int k = transpose_a == CblasTrans ? a_shape.At(num_axes - 2) : a_shape.At(num_axes - 1);
  return std::make_tuple(m, n, k);
}

}  // namespace

REGISTER_FUNCTION_CONFIG_DEF().Bool(
    "enable_float_compute_for_half_gemm", false,
    "true means that the type of intermedia value is float when compute half gemm");

template<DeviceType device_type, typename T>
class MatmulFloatingKernel final : public user_op::OpKernel {
 public:
  MatmulFloatingKernel() = default;
  ~MatmulFloatingKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CBLAS_TRANSPOSE trans_a = ctx->Attr<bool>("transpose_a") ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = ctx->Attr<bool>("transpose_b") ? CblasTrans : CblasNoTrans;
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(2, a->shape().NumAxes());

    int32_t m = 0, n = 0, k = 0;
    std::tie(m, n, k) = CalcMNK(a->shape(), out->shape(), trans_a);
    NewKernelUtil<device_type>::OFGemm(ctx->device_ctx(), trans_a, trans_b, m, n, k, GetOneVal<T>(),
                                       a->dptr<T>(), b->dptr<T>(), GetZeroVal<T>(),
                                       out->mut_dptr<T>());
  }
};

#define REGISTER_MATMUL_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("matmul")                           \
      .SetCreateFn<MatmulFloatingKernel<device, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value));

REGISTER_MATMUL_KERNEL(DeviceType::kCPU, float);
REGISTER_MATMUL_KERNEL(DeviceType::kCPU, double);
#ifdef WITH_CUDA
REGISTER_MATMUL_KERNEL(DeviceType::kGPU, float);
REGISTER_MATMUL_KERNEL(DeviceType::kGPU, double);
#endif

#ifdef WITH_CUDA
class MatmulGpuHalfKernel final : public user_op::OpKernel {
 public:
  MatmulGpuHalfKernel() = default;
  ~MatmulGpuHalfKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CBLAS_TRANSPOSE trans_a = ctx->Attr<bool>("transpose_a") ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = ctx->Attr<bool>("transpose_b") ? CblasTrans : CblasNoTrans;
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(2, a->shape().NumAxes());

    int32_t m = 0, n = 0, k = 0;
    std::tie(m, n, k) = CalcMNK(a->shape(), out->shape(), trans_a);

    if (ctx->job_desc().Bool("enable_float_compute_for_half_gemm")) {
      NewKernelUtil<DeviceType::kGPU>::OFHGemmWithFloat(
          ctx->device_ctx(), trans_a, trans_b, m, n, k, GetOneVal<float>(), a->dptr<float16>(),
          b->dptr<float16>(), GetZeroVal<float>(), out->mut_dptr<float16>());
    } else {
      NewKernelUtil<DeviceType::kGPU>::OFGemm(
          ctx->device_ctx(), trans_a, trans_b, m, n, k, GetOneVal<float16>(), a->dptr<float16>(),
          b->dptr<float16>(), GetZeroVal<float16>(), out->mut_dptr<float16>());
    }
  }
};
#endif

#ifdef WITH_CUDA
REGISTER_USER_KERNEL("matmul").SetCreateFn<MatmulGpuHalfKernel>().SetIsMatchedHob(
    (user_op::HobDeviceTag() == "gpu") & (user_op::HobDataType("a", 0) == DataType::kFloat16));
#endif

template<DeviceType device_type, typename T>
class BatchMatmulFloatingKernel final : public user_op::OpKernel {
 public:
  BatchMatmulFloatingKernel() = default;
  ~BatchMatmulFloatingKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CBLAS_TRANSPOSE trans_a = ctx->Attr<bool>("transpose_a") ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = ctx->Attr<bool>("transpose_b") ? CblasTrans : CblasNoTrans;
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* tmp_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int32_t num_axes = a->shape().NumAxes();
    CHECK_GT(num_axes, 2);

    int32_t m = 0, n = 0, k = 0;
    std::tie(m, n, k) = CalcMNK(a->shape(), out->shape(), trans_a);

    size_t batch_size = a->shape().Count(0, num_axes - 2);
    T** buf_dptr = reinterpret_cast<T**>(tmp_buf->mut_dptr<void>());
    NewKernelUtil<device_type>::OFBatchedGemm(ctx->device_ctx(), trans_a, trans_b, batch_size, m, n,
                                              k, GetOneVal<T>(), a->dptr<T>(), b->dptr<T>(),
                                              GetZeroVal<T>(), out->mut_dptr<T>(), buf_dptr);
  }
};

#define REGISTER_BATCH_MATMUL_KERNEL(device, dtype)                                   \
  REGISTER_USER_KERNEL("batch_matmul")                                                \
      .SetCreateFn<BatchMatmulFloatingKernel<device, dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                            \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                             \
        user_op::TensorDesc* a = ctx->TensorDesc4ArgNameAndIndex("a", 0);             \
        size_t num_axes = a->shape().NumAxes();                                       \
        size_t batch_num = a->shape().Count(0, num_axes - 2);                         \
        return sizeof(int64_t) * 3 * batch_num;                                       \
      })

REGISTER_BATCH_MATMUL_KERNEL(DeviceType::kCPU, float);
REGISTER_BATCH_MATMUL_KERNEL(DeviceType::kCPU, double);
#ifdef WITH_CUDA
REGISTER_BATCH_MATMUL_KERNEL(DeviceType::kGPU, float);
REGISTER_BATCH_MATMUL_KERNEL(DeviceType::kGPU, double);
#endif

#ifdef WITH_CUDA
class BatchMatmulGpuHalfKernel final : public user_op::OpKernel {
 public:
  BatchMatmulGpuHalfKernel() = default;
  ~BatchMatmulGpuHalfKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CBLAS_TRANSPOSE trans_a = ctx->Attr<bool>("transpose_a") ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = ctx->Attr<bool>("transpose_b") ? CblasTrans : CblasNoTrans;
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* tmp_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int32_t num_axes = a->shape().NumAxes();
    CHECK_GT(num_axes, 2);

    int32_t m = 0, n = 0, k = 0;
    std::tie(m, n, k) = CalcMNK(a->shape(), out->shape(), trans_a);

    size_t batch_size = a->shape().Count(0, num_axes - 2);
    float16** buf_dptr = reinterpret_cast<float16**>(tmp_buf->mut_dptr<void>());
    if (ctx->job_desc().Bool("enable_float_compute_for_half_gemm")) {
      NewKernelUtil<DeviceType::kGPU>::OFBatchedHGemmWithFloat(
          ctx->device_ctx(), trans_a, trans_b, batch_size, m, n, k, GetOneVal<float>(),
          a->dptr<float16>(), b->dptr<float16>(), GetZeroVal<float>(), out->mut_dptr<float16>(),
          buf_dptr);
    } else {
      NewKernelUtil<DeviceType::kGPU>::OFBatchedGemm(
          ctx->device_ctx(), trans_a, trans_b, batch_size, m, n, k, GetOneVal<float16>(),
          a->dptr<float16>(), b->dptr<float16>(), GetZeroVal<float16>(), out->mut_dptr<float16>(),
          buf_dptr);
    }
  }
};

REGISTER_USER_KERNEL("batch_matmul")
    .SetCreateFn<BatchMatmulGpuHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("a", 0) == DataType::kFloat16))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      user_op::TensorDesc* a = ctx->TensorDesc4ArgNameAndIndex("a", 0);
      size_t num_axes = a->shape().NumAxes();
      size_t batch_num = a->shape().Count(0, num_axes - 2);
      return sizeof(int64_t) * 3 * batch_num;
    });
#endif

}  // namespace oneflow
