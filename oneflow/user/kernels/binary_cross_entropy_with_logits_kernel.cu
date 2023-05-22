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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

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

template<typename INPUT_T, typename TARGET_T, WeightType WEIGHT_TYPE>
struct BinaryCrossEntropyWithLogitsFunctor;

template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kNone> {
  TARGET_T zero_;
  TARGET_T one_;
  BinaryCrossEntropyWithLogitsFunctor()
      : zero_(GetZeroVal<TARGET_T>()), one_(GetOneVal<TARGET_T>()) {}
  __device__ __forceinline__ TARGET_T operator()(INPUT_T input_val, TARGET_T target_val) const {
    const TARGET_T input_val_ = static_cast<TARGET_T>(input_val);
    const TARGET_T max_val = -input_val_ < zero_ ? zero_ : -input_val_;
    return (one_ - target_val) * input_val_ + max_val
           + (log(exp(-max_val) + exp(-input_val_ - max_val)));
  }
};

template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kPosWeight> {
  TARGET_T zero_;
  TARGET_T one_;
  BinaryCrossEntropyWithLogitsFunctor()
      : zero_(GetZeroVal<TARGET_T>()), one_(GetOneVal<TARGET_T>()) {}
  __device__ __forceinline__ TARGET_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                 TARGET_T weight_val) const {
    const TARGET_T input_val_ = static_cast<TARGET_T>(input_val);
    const TARGET_T max_val = -input_val_ < zero_ ? zero_ : -input_val_;
    const TARGET_T pos_weight_processed_val = weight_val - target_val + one_;
    return (one_ - target_val) * input_val_
           + (pos_weight_processed_val
              * (log(exp(-max_val) + exp(-input_val_ - max_val)) + max_val));
  }
};

template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, float, WeightType::kNone> {
  float zero_;
  float one_;
  BinaryCrossEntropyWithLogitsFunctor() : zero_(0.f), one_(1.f) {}
  __device__ __forceinline__ float operator()(INPUT_T input_val, float target_val) const {
    const float input_val_ = static_cast<float>(input_val);
    const float max_val = -input_val_ < zero_ ? zero_ : -input_val_;
    return (one_ - target_val) * input_val_ + max_val
           + (logf(expf(-max_val) + expf(-input_val_ - max_val)));
  }
};

template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, float, WeightType::kPosWeight> {
  float zero_;
  float one_;
  BinaryCrossEntropyWithLogitsFunctor() : zero_(0.f), one_(1.f) {}
  __device__ __forceinline__ float operator()(INPUT_T input_val, float target_val,
                                              float weight_val) const {
    const float input_val_ = static_cast<float>(input_val);
    const float max_val = -input_val_ < zero_ ? zero_ : -input_val_;
    const float pos_weight_processed_val = weight_val - target_val + one_;
    return (one_ - target_val) * input_val_
           + (pos_weight_processed_val
              * (logf(expf(-max_val) + expf(-input_val_ - max_val)) + max_val));
  }
};

template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kNone> f;
  __device__ __forceinline__ TARGET_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                 TARGET_T weight_val) const {
    return f(input_val, target_val) * weight_val;
  }
};

template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kPosWeight> f;
  __device__ __forceinline__ TARGET_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                 TARGET_T weight_val,
                                                 TARGET_T pos_weight_val) const {
    return f(input_val, target_val, pos_weight_val) * weight_val;
  }
};

template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, half, WeightType::kNone> {
  BinaryCrossEntropyWithLogitsFunctor<INPUT_T, float, WeightType::kNone> f;
  __device__ __forceinline__ half operator()(INPUT_T input_val, half target_val) const {
    return __float2half(f(input_val, __half2float(target_val)));
  }
};
template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, half, WeightType::kPosWeight> {
  BinaryCrossEntropyWithLogitsFunctor<INPUT_T, float, WeightType::kPosWeight> f;
  __device__ __forceinline__ half operator()(INPUT_T input_val, half target_val,
                                             half weight_val) const {
    return __float2half(f(input_val, __half2float(target_val), __half2float(weight_val)));
  }
};
template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, half, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsFunctor<INPUT_T, float, WeightType::kWeight> f;
  __device__ __forceinline__ half operator()(INPUT_T input_val, half target_val,
                                             half weight_val) const {
    return __float2half(f(input_val, __half2float(target_val), __half2float(weight_val)));
  }
};
template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsFunctor<INPUT_T, half, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsFunctor<INPUT_T, float, WeightType::kBoth> f;
  __device__ __forceinline__ half operator()(INPUT_T input_val, half target_val, half weight_val,
                                             half pos_weight_val) const {
    return __float2half(f(input_val, __half2float(target_val), __half2float(weight_val),
                          __half2float(pos_weight_val)));
  }
};

template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, half, WeightType::kNone> {
  BinaryCrossEntropyWithLogitsFunctor<float, float, WeightType::kNone> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val) const {
    return __float2half(f(__half2float(input_val), __half2float(target_val)));
  }
};
template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, half, WeightType::kPosWeight> {
  BinaryCrossEntropyWithLogitsFunctor<float, float, WeightType::kPosWeight> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val,
                                             half weight_val) const {
    return __float2half(
        f(__half2float(input_val), __half2float(target_val), __half2float(weight_val)));
  }
};
template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, half, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsFunctor<float, float, WeightType::kWeight> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val,
                                             half weight_val) const {
    return __float2half(
        f(__half2float(input_val), __half2float(target_val), __half2float(weight_val)));
  }
};
template<>
struct BinaryCrossEntropyWithLogitsFunctor<half, half, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsFunctor<float, float, WeightType::kBoth> f;
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

template<typename INPUT_T, typename TARGET_T, WeightType WEIGHT_TYPE>
struct BinaryCrossEntropyWithLogitsGradFunctor;

template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kNone> {
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                TARGET_T dy_val) const {
    return (CalSigmoid(input_val) - static_cast<INPUT_T>(target_val))
           * static_cast<INPUT_T>(dy_val);
  }
};
template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kPosWeight> {
  INPUT_T one_;
  BinaryCrossEntropyWithLogitsGradFunctor() : one_(GetOneVal<INPUT_T>()) {}
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                TARGET_T dy_val, TARGET_T weight_val) const {
    TARGET_T dx_tmp =
        dy_val
        * ((weight_val + one_ - target_val) * static_cast<TARGET_T>(CalSigmoid(input_val))
           - weight_val);
    return static_cast<INPUT_T>(dx_tmp);
  }
};
template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kNone> f;
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                TARGET_T dy_val, TARGET_T weight_val) const {
    return f(input_val, target_val, dy_val) * static_cast<INPUT_T>(weight_val);
  }
};
template<typename INPUT_T, typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kPosWeight> f;
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, TARGET_T target_val,
                                                TARGET_T dy_val, TARGET_T weight_val,
                                                TARGET_T pos_weight_val) const {
    return f(input_val, target_val, dy_val, pos_weight_val) * static_cast<INPUT_T>(weight_val);
  }
};

template<>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, half, WeightType::kNone> {
  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val) const {
    return (CalSigmoid(input_val) - target_val) * dy_val;
  }
};
template<>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, half, WeightType::kPosWeight> {
  half one_;
  BinaryCrossEntropyWithLogitsGradFunctor() : one_(GetOneVal<half>()) {}
  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val,
                                             half weight_val) const {
    return dy_val * ((weight_val + one_ - target_val) * CalSigmoid(input_val) - weight_val);
  }
};
template<>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, half, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsGradFunctor<half, half, WeightType::kNone> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val,
                                             half weight_val) const {
    return f(input_val, target_val, dy_val) * weight_val;
  }
};
template<>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, half, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsGradFunctor<half, half, WeightType::kPosWeight> f;
  __device__ __forceinline__ half operator()(half input_val, half target_val, half dy_val,
                                             half weight_val, half pos_weight_val) const {
    return f(input_val, target_val, dy_val, pos_weight_val) * weight_val;
  }
};

template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, half, WeightType::kNone> {
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, half target_val,
                                                half dy_val) const {
    return (CalSigmoid(input_val) - static_cast<INPUT_T>(__half2float(target_val)))
           * static_cast<INPUT_T>(__half2float(dy_val));
  }
};
template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, half, WeightType::kPosWeight> {
  INPUT_T one_;
  BinaryCrossEntropyWithLogitsGradFunctor() : one_(GetOneVal<INPUT_T>()) {}
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, half target_val, half dy_val,
                                                half weight_val) const {
    const INPUT_T dy_val_f = static_cast<INPUT_T>(__half2float(dy_val));
    const INPUT_T target_val_f = static_cast<INPUT_T>(__half2float(target_val));
    const INPUT_T weight_val_f = static_cast<INPUT_T>(__half2float(weight_val));
    return dy_val_f * ((weight_val_f + one_ - target_val_f) * CalSigmoid(input_val)) - weight_val_f;
  }
};
template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, half, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, half, WeightType::kNone> f;
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, half target_val, half dy_val,
                                                half weight_val) const {
    return f(input_val, target_val, dy_val) * static_cast<INPUT_T>(__half2float(weight_val));
  }
};
template<typename INPUT_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, half, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, half, WeightType::kPosWeight> f;
  __device__ __forceinline__ INPUT_T operator()(INPUT_T input_val, half target_val, half dy_val,
                                                half weight_val, half pos_weight_val) const {
    return f(input_val, target_val, dy_val, pos_weight_val)
           * static_cast<INPUT_T>(__half2float(weight_val));
  }
};

template<typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, TARGET_T, WeightType::kNone> {
  __device__ __forceinline__ half operator()(half input_val, TARGET_T target_val,
                                             TARGET_T dy_val) const {
    const half dy_val_h = __float2half(static_cast<float>(dy_val));
    const half target_val_h = __float2half(static_cast<float>(target_val));
    return (CalSigmoid(input_val) - target_val_h) * dy_val_h;
  }
};
template<typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, TARGET_T, WeightType::kPosWeight> {
  half one_;
  BinaryCrossEntropyWithLogitsGradFunctor() : one_(GetOneVal<half>()) {}
  __device__ __forceinline__ half operator()(half input_val, TARGET_T target_val, TARGET_T dy_val,
                                             TARGET_T weight_val) const {
    const half dy_val_h = __float2half(static_cast<float>(dy_val));
    const half target_val_h = __float2half(static_cast<float>(target_val));
    const half weight_val_h = __float2half(static_cast<float>(weight_val));
    return dy_val_h * ((weight_val_h + one_ - target_val_h) * CalSigmoid(input_val) - weight_val_h);
  }
};
template<typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, TARGET_T, WeightType::kWeight> {
  BinaryCrossEntropyWithLogitsGradFunctor<half, TARGET_T, WeightType::kNone> f;
  __device__ __forceinline__ half operator()(half input_val, TARGET_T target_val, TARGET_T dy_val,
                                             TARGET_T weight_val) const {
    return f(input_val, target_val, dy_val) * __float2half(static_cast<float>(weight_val));
  }
};
template<typename TARGET_T>
struct BinaryCrossEntropyWithLogitsGradFunctor<half, TARGET_T, WeightType::kBoth> {
  BinaryCrossEntropyWithLogitsGradFunctor<half, TARGET_T, WeightType::kPosWeight> f;
  __device__ __forceinline__ half operator()(half input_val, TARGET_T target_val, TARGET_T dy_val,
                                             TARGET_T weight_val, TARGET_T pos_weight_val) const {
    return f(input_val, target_val, dy_val, pos_weight_val)
           * __float2half(static_cast<float>(weight_val));
  }
};

template<typename INPUT_T, typename TARGET_T>
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

    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    TARGET_T* out = out_blob->mut_dptr<TARGET_T>();

    if (ctx->Attr<bool>("has_pos_weight")) {
      TARGET_T* pos_weight_processed = tmp_buffer_blob->mut_dptr<TARGET_T>();
      const TARGET_T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<TARGET_T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      auto bcast_mul =
          ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
              ctx->device_type(), ep::primitive::BinaryOp::kMul, target_blob->data_type(),
              target_blob->data_type(), target_blob->shape_view().NumAxes());
      CHECK(bcast_mul);
      bcast_mul->Launch(ctx->stream(), target_blob->shape_view().NumAxes(),
                        target_blob->shape_view().ptr(), target, pos_weight_shape.NumAxes(),
                        pos_weight_shape.dim_vec().data(), pos_weight, pos_weight_processed);
      if (ctx->has_input("weight", 0)) {
        const TARGET_T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<TARGET_T>();
        using FunctorT = BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kBoth>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK(
            (cuda::elementwise::
                 GenericLauncher<FactoryT, TARGET_T, INPUT_T, TARGET_T, TARGET_T, TARGET_T>::Launch(
                     FactoryT(FunctorT()), elem_cnt, out, input, target, weight,
                     pos_weight_processed, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      } else {
        OF_CUDA_CHECK((cuda::elementwise::Ternary(
            BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kPosWeight>(),
            elem_cnt, out, input, target, pos_weight_processed,
            ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    } else {
      if (ctx->has_input("weight", 0)) {
        const TARGET_T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<TARGET_T>();
        OF_CUDA_CHECK((cuda::elementwise::Ternary(
            BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kWeight>(), elem_cnt,
            out, input, target, weight, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      } else {
        OF_CUDA_CHECK((cuda::elementwise::Binary(
            BinaryCrossEntropyWithLogitsFunctor<INPUT_T, TARGET_T, WeightType::kNone>(), elem_cnt,
            out, input, target, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename INPUT_T, typename TARGET_T>
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

    const TARGET_T* dy = dy_blob->dptr<TARGET_T>();
    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    INPUT_T* dx = dx_blob->mut_dptr<INPUT_T>();

    if (ctx->Attr<bool>("has_pos_weight")) {
      TARGET_T* pos_weight_processed = tmp_buffer_blob->mut_dptr<TARGET_T>();
      const TARGET_T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<TARGET_T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      auto bcast_mul =
          ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
              ctx->device_type(), ep::primitive::BinaryOp::kMul, target_blob->data_type(),
              target_blob->data_type(), target_blob->shape_view().NumAxes());
      CHECK(bcast_mul);
      bcast_mul->Launch(ctx->stream(), target_blob->shape_view().NumAxes(),
                        target_blob->shape_view().ptr(), target, pos_weight_shape.NumAxes(),
                        pos_weight_shape.dim_vec().data(), pos_weight, pos_weight_processed);
      if (ctx->has_input("weight", 0)) {
        const TARGET_T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<TARGET_T>();
        using FunctorT =
            BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kBoth>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK((cuda::elementwise::GenericLauncher<
                       FactoryT, INPUT_T, INPUT_T, TARGET_T, TARGET_T, TARGET_T,
                       TARGET_T>::Launch(FactoryT(FunctorT()), elem_cnt, dx, input, target, dy,
                                         weight, pos_weight_processed,
                                         ctx->stream()->As<ep::CudaStream>()->cuda_stream())));

      } else {
        using FunctorT =
            BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kPosWeight>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK(
            (cuda::elementwise::
                 GenericLauncher<FactoryT, INPUT_T, INPUT_T, TARGET_T, TARGET_T, TARGET_T>::Launch(
                     FactoryT(FunctorT()), elem_cnt, dx, input, target, dy, pos_weight_processed,
                     ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      }
    } else {
      if (ctx->has_input("weight", 0)) {
        const TARGET_T* weight = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<TARGET_T>();
        using FunctorT =
            BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kWeight>;
        using FactoryT = cuda::elementwise::SimpleFactory<FunctorT>;
        OF_CUDA_CHECK(
            (cuda::elementwise::
                 GenericLauncher<FactoryT, INPUT_T, INPUT_T, TARGET_T, TARGET_T, TARGET_T>::Launch(
                     FactoryT(FunctorT()), elem_cnt, dx, input, target, dy, weight,
                     ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
      } else {
        OF_CUDA_CHECK((cuda::elementwise::Ternary(
            BinaryCrossEntropyWithLogitsGradFunctor<INPUT_T, TARGET_T, WeightType::kNone>(),
            elem_cnt, dx, input, target, dy, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
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

#define REGISTER_BINARY_CROSS_ENTROPY_KERNEL(input_dtype, target_dtype)                           \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits")                                        \
      .SetCreateFn<BinaryCrossEntropyWithLogitsKernel<input_dtype, target_dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("input", 0) == GetDataType<input_dtype>::value)   \
                       && (user_op::HobDataType("target", 0) == GetDataType<target_dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<target_dtype>::value))   \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<target_dtype>());

#define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(input_dtype, target_dtype)                      \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_grad")                                   \
      .SetCreateFn<BinaryCrossEntropyWithLogitsGradKernel<input_dtype, target_dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("input", 0) == GetDataType<input_dtype>::value)   \
                       && (user_op::HobDataType("target", 0) == GetDataType<target_dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<target_dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<input_dtype>::value))     \
      .SetInferTmpSizeFn(GenBwInferTmpSizeFn<target_dtype>());

REGISTER_BINARY_CROSS_ENTROPY_KERNEL(half, half)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(half, float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float, half)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(half, double)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double, half)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float, float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float, double)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double, float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double, double)

REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half, half)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half, float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float, half)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half, double)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double, half)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float, float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float, double)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double, float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double, double)

}  // namespace user_op
}  // namespace oneflow
