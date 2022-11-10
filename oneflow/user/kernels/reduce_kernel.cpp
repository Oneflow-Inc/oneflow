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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"

#ifdef WITH_CUDA
#include "oneflow/core/ep/cuda/cuda_device.h"
#endif  // WITH_CUDA
#include "oneflow/core/ep/include/primitive/matmul.h"

namespace oneflow {

namespace {

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewReduceMatmulTransAPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("input_tensor", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/true,
                            /*transpose_b=*/false);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewReduceMatmulNoTransAPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("input_tensor", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/false,
                            /*transpose_b=*/false);
}

template<typename Context>
std::unique_ptr<ep::primitive::Fill> NewFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("output_tensor", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->device_type(), data_type);
}

auto ReduceMatmulTransAPrimitiveExists() {
  return hob::make_custom("ReduceMatmulTransAPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewReduceMatmulTransAPrimitive(&ctx).operator bool();
                          });
}

auto ReduceMatmulNoTransAPrimitiveExists() {
  return hob::make_custom("ReduceMatmulNoTransAPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewReduceMatmulNoTransAPrimitive(&ctx).operator bool();
                          });
}

auto FillPrimitiveExists() {
  return hob::make_custom("FillPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewFillPrimitive(&ctx).operator bool();
  });
}

template<template<typename> class BinaryFunc, DeviceType device_type, typename T, typename K>
class ReduceKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ReduceKernel() = default;
  ~ReduceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    const int32_t output_elem_cnt = output_tensor->shape_view().elem_cnt();

    if (input_tensor->shape_view().elem_cnt() == 0) {
      if (output_tensor->shape_view().elem_cnt() != 0) {
        Scalar init_value = [&]() {
          if (std::is_same<BinaryFunc<T>, BinaryFuncAny<T>>::value) { return Scalar(0); }
          if (std::is_same<BinaryFunc<T>, BinaryFuncAll<T>>::value) { return Scalar(1); }
          return Scalar(0);
        }();
        CHECK_GE(output_elem_cnt, 0);
        if (output_elem_cnt == 0) { return; }
        std::unique_ptr<ep::primitive::Fill> fill = NewFillPrimitive(ctx);
        CHECK(fill);
        fill->Launch(ctx->stream(), output_tensor->mut_dptr<K>(), init_value, output_elem_cnt);
      }
      return;
    }
    const Shape& reduced_shape =
        CreateReducedShape(input_tensor->shape_view(), {axis.begin(), axis.end()});
    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->stream(), XpuVarNdarray<K>(reduced_shape, output_tensor->mut_dptr<K>()),
        XpuVarNdarray<const T>(input_tensor->shape_view(), input_tensor->dptr<T>()),
        XpuVarNdarray<T>(tmp_buffer->shape_view(), tmp_buffer->mut_dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_REDUCE_XPU_KERNEL(op_name, binary_func, device, dtype)                            \
  REGISTER_USER_KERNEL(op_name)                                                                    \
      .SetCreateFn<ReduceKernel<binary_func, device, dtype, dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                        \
                       && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const Shape& in_shape = ctx->InputShape("input_tensor", 0);                                \
        return in_shape.elem_cnt() * sizeof(dtype);                                                \
      });

#define REGISTER_REDUCE_LOGICAL_XPU_KERNEL(op_name, binary_func, device, dtype)                  \
  REGISTER_USER_KERNEL(op_name)                                                                  \
      .SetCreateFn<ReduceKernel<binary_func, device, dtype, bool>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("input_tensor", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("output_tensor", 0) == DataType::kBool)          \
                       && FillPrimitiveExists())                                                 \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape& in_shape = ctx->InputShape("input_tensor", 0);                              \
        return in_shape.elem_cnt() * sizeof(dtype);                                              \
      });

#define REGISTER_REDUCE_ARITHMETIC_KERNELS(device, dtype)                  \
  REGISTER_REDUCE_XPU_KERNEL("reduce_prod", BinaryFuncProd, device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_min", BinaryFuncMin, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_max", BinaryFuncMax, device, dtype)

#define REGISTER_REDUCE_NANSUM_KERNELS(device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_nansum", BinaryFuncNanSum, device, dtype)

#define REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(device) \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, bool)           \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, float)          \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, double)         \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int8_t)         \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, uint8_t)        \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int32_t)        \
  REGISTER_REDUCE_ARITHMETIC_KERNELS(device, int64_t)

#define REGISTER_REDUCE_NANSUM_KERNELS_BY_DEVICE(device) \
  REGISTER_REDUCE_NANSUM_KERNELS(device, float)          \
  REGISTER_REDUCE_NANSUM_KERNELS(device, double)

REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(DeviceType::kCPU)
REGISTER_REDUCE_NANSUM_KERNELS_BY_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_ARITHMETIC_KERNELS_BY_DEVICE(DeviceType::kCUDA)
REGISTER_REDUCE_NANSUM_KERNELS_BY_DEVICE(DeviceType::kCUDA)
#endif

#define REGISTER_REDUCE_SUM_KERNELS(device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_sum", BinaryFuncSum, device, dtype)

#define REGISTER_REDUCE_SUM_KERNELS_BY_DEVICE(device) \
  REGISTER_REDUCE_SUM_KERNELS(device, double)         \
  REGISTER_REDUCE_SUM_KERNELS(device, int8_t)         \
  REGISTER_REDUCE_SUM_KERNELS(device, uint8_t)        \
  REGISTER_REDUCE_SUM_KERNELS(device, int32_t)        \
  REGISTER_REDUCE_SUM_KERNELS(device, int64_t)

REGISTER_REDUCE_SUM_KERNELS_BY_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_SUM_KERNELS_BY_DEVICE(DeviceType::kCUDA)
#endif
REGISTER_REDUCE_SUM_KERNELS(DeviceType::kCPU, float)
REGISTER_REDUCE_SUM_KERNELS(DeviceType::kCPU, float16)

#define REGISTER_REDUCE_LOGICAL_KERNELS(device)                                    \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, bool)    \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, bool)    \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, float)   \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, float)   \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, double)  \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, double)  \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, int8_t)  \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, int8_t)  \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, uint8_t) \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, uint8_t) \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, int32_t) \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, int32_t) \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_any", BinaryFuncAny, device, int64_t) \
  REGISTER_REDUCE_LOGICAL_XPU_KERNEL("reduce_all", BinaryFuncAll, device, int64_t)

REGISTER_REDUCE_LOGICAL_KERNELS(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_REDUCE_LOGICAL_KERNELS(DeviceType::kCUDA)

namespace {

std::vector<int32_t> RegularAxis(const std::vector<int32_t>& axis) {
  std::vector<int32_t> regular_axis = axis;
  std::sort(regular_axis.begin(), regular_axis.end());
  return regular_axis;
}

void GetReduceSumLayout(const std::vector<int32_t>& axis, const ShapeView& in_shape,
                        bool* is_axis_contiguous, int64_t* outer_size, int64_t* inner_size,
                        int64_t* reduce_size) {
  if (!axis.empty()) {
    *is_axis_contiguous = ((axis.back() - axis.front() + 1) == axis.size());
    *outer_size = in_shape.Count(0, axis.front());
    *inner_size = in_shape.Count(axis.back() + 1);
    *reduce_size = in_shape.Count(axis.front(), axis.back() + 1);
  }
}

}  // namespace

class ReduceSumHalfKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ReduceSumHalfKernel() = default;
  ~ReduceSumHalfKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<int32_t> axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& in_shape = input_tensor->shape_view();
    const DataType data_type = input_tensor->data_type();
    bool is_axis_contiguous = false;
    int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
    GetReduceSumLayout(axis, in_shape, &is_axis_contiguous, &outer_size, &inner_size, &reduce_size);
    if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {
      bool trans_a = (inner_size != 1);
      const int32_t m = (inner_size == 1) ? outer_size : inner_size;
      const int32_t n = 1;
      const int32_t k = reduce_size;
      const void* ones = nullptr;
      auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
      if (cuda_device != nullptr) { ones = cuda_device->GetConstOnes(data_type, reduce_size); }
      if (ones == nullptr) {
        std::unique_ptr<ep::primitive::Fill> fill =
            ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                                    data_type);
        CHECK(fill);
        fill->Launch(ctx->stream(), tmp_buffer->mut_dptr(), 1.0, reduce_size);
        ones = tmp_buffer->dptr();
      }
      std::unique_ptr<ep::primitive::Matmul> matmul;
      if (trans_a) {
        matmul = NewReduceMatmulTransAPrimitive(ctx);
      } else {
        matmul = NewReduceMatmulNoTransAPrimitive(ctx);
      }
      matmul->Launch(ctx->stream(), m, n, k, 1.0, input_tensor->dptr(), ones, 0.0,
                     output_tensor->mut_dptr());
    } else {
      const Shape& reduced_shape = CreateReducedShape(in_shape, {axis.begin(), axis.end()});
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
               tmp_buffer->shape_view().elem_cnt());
      auto h2f = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
          ctx->device_type(), data_type, DataType::kFloat);
      CHECK(h2f);
      auto f2h = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
          ctx->device_type(), DataType::kFloat, data_type);
      CHECK(f2h);
      h2f->Launch(ctx->stream(), input_tensor->dptr(), in_tmp_buffer, in_shape.elem_cnt());

      NdarrayReduce<DeviceType::kCUDA, float, BinaryFuncSum>::Reduce(
          ctx->stream(), XpuVarNdarray<float>(reduced_shape, out_tmp_buffer),
          XpuVarNdarray<const float>(in_shape, in_tmp_buffer),
          XpuVarNdarray<float>(in_shape, reduce_tmp_buffer));

      f2h->Launch(ctx->stream(), out_tmp_buffer, output_tensor->mut_dptr(),
                  output_tensor->shape_view().elem_cnt());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_SUM_HALF_KERNEL(dtype)                                                    \
  REGISTER_USER_KERNEL("reduce_sum")                                                              \
      .SetCreateFn<ReduceSumHalfKernel>()                                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("output_tensor", 0) == GetDataType<dtype>::value) \
                       && ReduceMatmulTransAPrimitiveExists()                                     \
                       && ReduceMatmulNoTransAPrimitiveExists())                                  \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape& in_shape = ctx->InputTensorDesc("input_tensor", 0).shape();                  \
        const Shape& out_shape = ctx->OutputTensorDesc("output_tensor", 0).shape();               \
        const auto& axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));                  \
        bool is_axis_contiguous = false;                                                          \
        int64_t outer_size = 0, inner_size = 0, reduce_size = 0;                                  \
        GetReduceSumLayout(axis, ShapeView(in_shape), &is_axis_contiguous, &outer_size,           \
                           &inner_size, &reduce_size);                                            \
        size_t tmp_bytes = 0;                                                                     \
        if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {                         \
          tmp_bytes = GetCudaAlignedSize(reduce_size * sizeof(dtype));                            \
        } else {                                                                                  \
          tmp_bytes = (2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float))                \
                       + GetCudaAlignedSize(out_shape.elem_cnt() * sizeof(float)));               \
        }                                                                                         \
        return tmp_bytes;                                                                         \
      });

REGISTER_REDUCE_SUM_HALF_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_REDUCE_SUM_HALF_KERNEL(nv_bfloat16)
#endif

class ReduceSumFloatCudaKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ReduceSumFloatCudaKernel() = default;
  ~ReduceSumFloatCudaKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<int32_t> axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ShapeView& in_shape = input_tensor->shape_view();
    if (input_tensor->shape_view().elem_cnt() == 0) {
      if (output_tensor->shape_view().elem_cnt() != 0) {
        Memset<DeviceType::kCUDA>(
            ctx->stream(), output_tensor->mut_dptr<float>(), 0,
            output_tensor->shape_view().elem_cnt() * GetSizeOfDataType(output_tensor->data_type()));
      }
      return;
    }
    bool is_axis_contiguous = false;
    int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
    GetReduceSumLayout(axis, in_shape, &is_axis_contiguous, &outer_size, &inner_size, &reduce_size);
    const float* ones = nullptr;
    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    if (cuda_device != nullptr) {
      ones = static_cast<const float*>(cuda_device->GetConstOnes(DataType::kFloat, reduce_size));
    }
    if ((!axis.empty()) && in_shape.NumAxes() > 0 && is_axis_contiguous
        && (outer_size == 1 || inner_size == 1) && ones != nullptr
        && ParseBooleanFromEnv("ONEFLOW_KERNEL_REDUCE_SUM_USE_MATMUL", false)) {
      ep::primitive::BlasTransposeType trans_a = (inner_size == 1)
                                                     ? ep::primitive::BlasTransposeType::N
                                                     : ep::primitive::BlasTransposeType::T;
      ep::primitive::BlasTransposeType trans_b = ep::primitive::BlasTransposeType::N;
      const int32_t m = (inner_size == 1) ? outer_size : inner_size;
      const int32_t n = 1;
      const int32_t k = reduce_size;
#if CUDA_VERSION >= 11000
      CublasMathModeGuard guard(ctx->stream()->As<ep::CudaStream>()->cublas_handle());
      // disable tf32
      guard.SetMathMode(CUBLAS_DEFAULT_MATH);
#endif  // defined(WITH_CUDA) && CUDA_VERSION >= 11000
      auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
          DeviceType::kCUDA, DataType::kFloat, trans_a, trans_b);
      CHECK(matmul);
      matmul->Launch(ctx->stream(), m, n, k, 1.0, input_tensor->dptr(), ones, 0.0,
                     output_tensor->mut_dptr());
    } else {
      const Shape& reduced_shape = CreateReducedShape(in_shape, {axis.begin(), axis.end()});
      NdarrayReduce<DeviceType::kCUDA, float, BinaryFuncSum>::Reduce(
          ctx->stream(), XpuVarNdarray<float>(reduced_shape, output_tensor->mut_dptr<float>()),
          XpuVarNdarray<const float>(input_tensor->shape_view(), input_tensor->dptr<float>()),
          XpuVarNdarray<float>(tmp_buffer->shape_view(), tmp_buffer->mut_dptr<float>()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("reduce_sum")
    .SetCreateFn<ReduceSumFloatCudaKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("output_tensor", 0) == DataType::kFloat))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const Shape& in_shape = ctx->InputTensorDesc("input_tensor", 0).shape();
      return GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
    });

#endif  // WITH_CUDA

}  // namespace oneflow
