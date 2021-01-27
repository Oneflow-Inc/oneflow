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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace {

template<template<typename> class BinaryFunc, DeviceType device_type, typename T>
class ReduceKernel final : public user_op::OpKernel {
 public:
  ReduceKernel() = default;
  ~ReduceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        CreateReducedShape(input_tensor->shape(), {axis.begin(), axis.end()});
    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, output_tensor->mut_dptr<T>()),
        XpuVarNdarray<const T>(input_tensor->shape(), input_tensor->dptr<T>()),
        XpuVarNdarray<T>(tmp_buffer->shape(), tmp_buffer->mut_dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_REDUCE_XPU_KERNEL(op_name, binary_func, device, dtype)                           \
  REGISTER_USER_KERNEL(op_name)                                                                   \
      .SetCreateFn<ReduceKernel<binary_func, device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                        \
                       & (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("input_tensor", 0);                    \
        return in_shape->elem_cnt() * sizeof(dtype);                                              \
      });

#define REGISTER_REDUCE_ARITHMETIC_KERNELS(device, dtype)                  \
  REGISTER_REDUCE_XPU_KERNEL("reduce_prod", BinaryFuncProd, device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_min", BinaryFuncMin, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_sum", BinaryFuncSum, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_max", BinaryFuncMax, device, dtype)

#define REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(device) \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, float)          \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, double)         \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int8_t)         \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int32_t)        \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int64_t)

REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(DeviceType::kGPU)
#endif

#define REGISTER_REDUCE_LOGICAL_KERNELS(device)                           \
  REGISTER_REDUCE_XPU_KERNEL("reduce_any", BinaryFuncAny, device, int8_t) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_all", BinaryFuncAll, device, int8_t)

REGISTER_REDUCE_LOGICAL_KERNELS(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_LOGICAL_KERNELS(DeviceType::kGPU)

namespace {

std::vector<int32_t> RegularAxis(const std::vector<int32_t>& axis) {
  std::vector<int32_t> regular_axis = axis;
  std::sort(regular_axis.begin(), regular_axis.end());
  return regular_axis;
}

void GetReduceSumLayout(const std::vector<int32_t>& axis, const ShapeView& in_shape,
                        bool* is_axis_contiguous, int64_t* outer_size, int64_t* inner_size,
                        int64_t* reduce_size) {
  *is_axis_contiguous = ((axis.back() - axis.front() + 1) == axis.size());
  *outer_size = in_shape.Count(0, axis.front());
  *inner_size = in_shape.Count(axis.back() + 1);
  *reduce_size = in_shape.Count(axis.front(), axis.back() + 1);
}

}  // namespace

class ReduceSumHalfKernel final : public user_op::OpKernel {
 public:
  explicit ReduceSumHalfKernel(user_op::KernelCreateContext* ctx) {
    axis_ = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
  }
  ~ReduceSumHalfKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& in_shape = input_tensor->shape();
    bool is_axis_contiguous = false;
    int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
    GetReduceSumLayout(axis_, in_shape, &is_axis_contiguous, &outer_size, &inner_size,
                       &reduce_size);
    if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {
      CBLAS_TRANSPOSE trans_a = (inner_size == 1) ? CblasNoTrans : CblasTrans;
      CBLAS_TRANSPOSE trans_b = CblasNoTrans;
      const int32_t m = (inner_size == 1) ? outer_size : inner_size;
      const int32_t n = 1;
      const int32_t k = reduce_size;
      NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), reduce_size,
                                            static_cast<float16>(1.0),
                                            tmp_buffer->mut_dptr<float16>());
      NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), trans_a, trans_b, m, n, k,
                                              GetOneVal<float16>(), input_tensor->dptr<float16>(),
                                              tmp_buffer->dptr<float16>(), GetZeroVal<float16>(),
                                              output_tensor->mut_dptr<float16>());
    } else {
      const Shape& reduced_shape = CreateReducedShape(in_shape, {axis_.begin(), axis_.end()});
      float* in_tmp_buffer = tmp_buffer->mut_dptr<float>();
      const size_t in_tmp_buffer_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
      float* out_tmp_buffer =
          reinterpret_cast<float*>(tmp_buffer->mut_dptr<char>() + in_tmp_buffer_bytes);
      const size_t out_tmp_buffer_bytes =
          GetCudaAlignedSize(reduced_shape.elem_cnt() * sizeof(float));
      float* reduce_tmp_buffer = reinterpret_cast<float*>(
          tmp_buffer->mut_dptr<char>() + in_tmp_buffer_bytes + out_tmp_buffer_bytes);
      const size_t reduce_tmp_buffer_bytes =
          GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
      CHECK_LE(in_tmp_buffer_bytes + out_tmp_buffer_bytes + reduce_tmp_buffer_bytes,
               tmp_buffer->shape().elem_cnt());
      CopyElemOnGpu<float16, float>(ctx->device_ctx(), input_tensor->dptr<float16>(), in_tmp_buffer,
                                    in_shape.elem_cnt());

      NdarrayReduce<DeviceType::kGPU, float, BinaryFuncSum>::Reduce(
          ctx->device_ctx(), XpuVarNdarray<float>(reduced_shape, out_tmp_buffer),
          XpuVarNdarray<const float>(in_shape, in_tmp_buffer),
          XpuVarNdarray<float>(in_shape, reduce_tmp_buffer));

      CopyElemOnGpu<float, float16>(ctx->device_ctx(), out_tmp_buffer,
                                    output_tensor->mut_dptr<float16>(),
                                    output_tensor->shape().elem_cnt());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  std::vector<int32_t> axis_;
};

REGISTER_USER_KERNEL("reduce_sum")
    .SetCreateWithCtxFn<ReduceSumHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("output_tensor", 0) == GetDataType<float16>::value))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const Shape& in_shape = ctx->TensorDesc4ArgNameAndIndex("input_tensor", 0)->shape();
      const Shape& out_shape = ctx->TensorDesc4ArgNameAndIndex("output_tensor", 0)->shape();
      const auto& axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
      bool is_axis_contiguous = false;
      int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
      GetReduceSumLayout(axis, ShapeView(in_shape), &is_axis_contiguous, &outer_size, &inner_size,
                         &reduce_size);
      size_t tmp_bytes = 0;
      if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {
        tmp_bytes = GetCudaAlignedSize(reduce_size * sizeof(float16));
      } else {
        tmp_bytes = (2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float))
                     + GetCudaAlignedSize(out_shape.elem_cnt() * sizeof(float)));
      }
      return tmp_bytes;
    });

#endif

}  // namespace oneflow
