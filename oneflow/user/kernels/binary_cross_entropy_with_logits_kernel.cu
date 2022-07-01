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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

enum class WeightType {
  kNone,
  kWeight,
  kPosWeight,
  kBoth,
};

template<typename T, WeightType WEIGHT_TYPE>
struct BinaryCrossEntropyWithLogitsFunctor;

template<typename T>
struct BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kNone> {
  T zero_;
  T one_;
  BinaryCrossEntropyWithLogitsFunctor() : zero_(GetZeroVal<T>()), one_(GetOneVal<T>()) {}
  __device__ __forceinline__ T operator()(T input_val, T target_val) const {
    const T max_val = -input_val < zero_ ? zero_ : -input_val;
    return (one_ - target_val) * input_val + max_val
           + (log(exp(-max_val) + exp(-input_val - max_val)));
  }
};

template<typename T>
struct BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kPosWeight> {
  T zero_;
  T one_;
  BinaryCrossEntropyWithLogitsFunctor() : zero_(GetZeroVal<T>()), one_(GetOneVal<T>()) {}
  __device__ __forceinline__ T operator()(T input_val, T target_val, T weight_val) const {
    const T max_val = -input_val < zero_ ? zero_ : -input_val;
    const T pos_weight_processed_val = weight_val - target_val + one_;
    return (one_ - target_val) * input_val
           + (pos_weight_processed_val
              * (log(exp(-max_val) + exp(-input_val - max_val)) + max_val));
  }
};

template<>
struct BinaryCrossEntropyWithLogitsFunctor<float, WeightType::kNone> {
  float zero_;
  float one_;
  BinaryCrossEntropyWithLogitsFunctor() : zero_(0.f), one_(1.f) {}
  __device__ __forceinline__ float operator()(float input_val, float target_val) const {
    const float max_val = -input_val < zero_ ? zero_ : -input_val;
    return (one_ - target_val) * input_val + max_val
           + (logf(expf(-max_val) + expf(-input_val - max_val)));
  }
};

template<>
struct BinaryCrossEntropyWithLogitsFunctor<float, WeightType::kPosWeight> {
  float zero_;
  float one_;
  BinaryCrossEntropyWithLogitsFunctor() : zero_(0.f), one_(1.f) {}
  __device__ __forceinline__ float operator()(float input_val, float target_val,
                                              float weight_val) const {
    const float max_val = -input_val < zero_ ? zero_ : -input_val;
    const float pos_weight_processed_val = weight_val - target_val + one_;
    return (one_ - target_val) * input_val
           + (pos_weight_processed_val
              * (logf(expf(-max_val) + expf(-input_val - max_val)) + max_val));
  }
};

template<typename T>
struct BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kNone> f;
  __device__ __forceinline__ T operator()(T input_val, T target_val, T weight_val) const {
    return f(input_val, target_val) * weight_val;
  }
};

template<typename T>
struct BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kPosWeight> f;
  __device__ __forceinline__ T operator()(T input_val, T target_val, T weight_val,
                                          T pos_weight_val) const {
    return f(input_val, target_val, pos_weight_val) * weight_val;
  }
};

template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, WeightType::kNone> {
  BinaryCrossEntropyWithLogitsFunctor<float, WeightType::kNone> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val) const {
    return __float2half(f(__half2float(input_val), __half2float(target_val)));
  }
};
template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, WeightType::kPosWeight> {
  BinaryCrossEntropyWithLogitsFunctor<float, WeightType::kPosWeight> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val,
                                             half weight_val) const {
    return __float2half(
        f(__half2float(input_val), __half2float(target_val), __half2float(weight_val)));
  }
};
template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsFunctor<float, WeightType::kWeight> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val,
                                             half weight_val) const {
    return __float2half(
        f(__half2float(input_val), __half2float(target_val), __half2float(weight_val)));
  }
};
template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsFunctor<float, WeightType::kBoth> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val, half weight_val,
                                             half pos_weight_val) const {
    return __float2half(f(__half2float(input_val), __half2float(target_val),
                          __half2float(weight_val), __half2float(pos_weight_val)));
  }
};

template<typename T>
__device__ __forceinline__ T CalSigmoid(const T x) {
  const T half_of_one = static_cast<T>(0.5);
  return half_of_one * tanh(half_of_one * x) + half_of_one;
}

template<>
__device__ __forceinline__ float CalSigmoid(const float x) {
  const float half_of_one = static_cast<float>(0.5);
  return half_of_one * tanhf(half_of_one * x) + half_of_one;
}

template<>
__device__ __forceinline__ half CalSigmoid(const half x) {
  return __float2half(CalSigmoid(__half2float(x)));
}

template<typename T, WeightType WEIGHT_TYPE>
struct BinaryCrossEntropyWithLogitsGradFunctor;

template<typename T>
struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kNone> {
  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val) const {
    return (CalSigmoid(input_val) - target_val) * dy_val;
  }
};

template<typename T>
struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kPosWeight> {
  T one_;
  BinaryCrossEntropyWithLogitsGradFunctor() : one_(GetOneVal<T>()) {}
  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val) const {
    return dy_val * ((weight_val + one_ - target_val) * CalSigmoid(input_val) - weight_val);
  }
};

template<typename T>
struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kNone> f;
  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val) const {
    return f(input_val, target_val, dy_val) * weight_val;
  }
};

template<typename T>
struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kPosWeight> f;
  __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val,
                                          T pos_weight_val) const {
    return f(input_val, target_val, dy_val, pos_weight_val) * weight_val;
  }
};

template<typename T>
class BinaryCrossEntropyWithLogitsKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsKernel() = default;
  ~BinaryCrossEntropyWithLogitsKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();

    if (ctx->Attr<bool>("has_pos_weight")) {
      T* pos_weight_processed = tmp_buffer_blob->mut_dptr<T>();
      const T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      NdarrayUtil<DeviceType::kCUDA, T>::BroadcastMul(
          ctx->stream(), XpuVarNdarray<T>(target_blob->shape_view(), pos_weight_processed),
          XpuVarNdarray<const T>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const T>(target_blob->shape_view(), target));
      if (ctx->has_input("weight", 0)) {
        const T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
        using FunctorT = BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kBoth>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK((cuda::elementwise::GenericLauncher<FactoryT, T, T, T, T, T>::Launch(
            FactoryT(FunctorT()), elem_cnt, out, input, target, weight, pos_weight_processed,
            ctx->stream()->As<ep::CudaStream>()->cuda_stream())));

      } else {
        OF_CUDA_CHECK((cuda::elementwise::Ternary(
            BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kPosWeight>(), elem_cnt, out, input,
            target, pos_weight_processed, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    } else {
      if (ctx->has_input("weight", 0)) {
        const T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
        OF_CUDA_CHECK((cuda::elementwise::Ternary(
            BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kWeight>(), elem_cnt, out, input,
            target, weight, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      } else {
        OF_CUDA_CHECK((cuda::elementwise::Binary(
            BinaryCrossEntropyWithLogitsFunctor<T, WeightType::kNone>(), elem_cnt, out, input,
            target, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class BinaryCrossEntropyWithLogitsGradKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsGradKernel() = default;
  ~BinaryCrossEntropyWithLogitsGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();

    if (ctx->Attr<bool>("has_pos_weight")) {
      T* pos_weight_processed = tmp_buffer_blob->mut_dptr<T>();
      const T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      NdarrayUtil<DeviceType::kCUDA, T>::BroadcastMul(
          ctx->stream(), XpuVarNdarray<T>(target_blob->shape_view(), pos_weight_processed),
          XpuVarNdarray<const T>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const T>(target_blob->shape_view(), target));

      if (ctx->has_input("weight", 0)) {
        const T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
        using FunctorT = BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kBoth>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK((cuda::elementwise::GenericLauncher<FactoryT, T, T, T, T, T, T>::Launch(
            FactoryT(FunctorT()), elem_cnt, dx, input, target, dy, weight, pos_weight_processed,
            ctx->stream()->As<ep::CudaStream>()->cuda_stream())));

      } else {
        using FunctorT = BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kPosWeight>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK((cuda::elementwise::GenericLauncher<FactoryT, T, T, T, T, T>::Launch(
            FactoryT(FunctorT()), elem_cnt, dx, input, target, dy, pos_weight_processed,
            ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    } else {
      if (ctx->has_input("weight", 0)) {
        const T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
        using FunctorT = BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kWeight>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK((cuda::elementwise::GenericLauncher<FactoryT, T, T, T, T, T>::Launch(
            FactoryT(FunctorT()), elem_cnt, dx, input, target, dy, weight,
            ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      } else {
        OF_CUDA_CHECK((cuda::elementwise::Ternary(
            BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kNone>(), elem_cnt, dx, input,
            target, dy, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape("input", 0).elem_cnt();
    size_t tmp_buffer_size = 0;
    if (ctx->Attr<bool>("has_pos_weight")) { tmp_buffer_size += GetCudaAlignedSize(n * sizeof(T)); }
    return tmp_buffer_size;
  };
}
template<typename T>
user_op::InferTmpSizeFn GenBwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape("target", 0).elem_cnt();
    size_t tmp_buffer_size = 0;
    if (ctx->Attr<bool>("has_pos_weight")) { tmp_buffer_size += GetCudaAlignedSize(n * sizeof(T)); }
    return tmp_buffer_size;
  };
}

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits")                                 \
      .SetCreateFn<BinaryCrossEntropyWithLogitsKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<dtype>());

#define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_grad")                            \
      .SetCreateFn<BinaryCrossEntropyWithLogitsGradKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn(GenBwInferTmpSizeFn<dtype>());

REGISTER_BINARY_CROSS_ENTROPY_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double)

REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
