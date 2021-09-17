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
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T>
__global__ void ComputeBinaryCrossEntropyWithLogitsOut(int64_t elem_cnt, const T* input,
                                                       const T* target, T* out, const T* weight,
                                                       const T* pos_weight_processed) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    assert(input_val >= 0.0);
    assert(input_val <= 1.0);
    T max_val = -input_val < 0 ? 0 : -input_val;

    if (pos_weight_processed == nullptr) {
      out[i] =
          (1 - target_val) * input_val + max_val + (log(exp(-max_val) + exp(-input_val - max_val)));
    } else {
      T pos_weight_processed_val = pos_weight_processed[i] - target_val + 1;
      out[i] =
          (1 - target_val) * input_val
          + (pos_weight_processed_val * (log(exp(-max_val) + exp(-input_val - max_val)) + max_val));
    }
    if (weight != nullptr) { out[i] *= weight[i]; }
  }
}

__global__ void ComputeBinaryCrossEntropyWithLogitsOutHalf(int64_t elem_cnt, const half* input,
                                                           const half* target, half* out,
                                                           const half* weight,
                                                           const half* pos_weight_processed) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    float input_val = __half2float(input[i]);
    float target_val = __half2float(target[i]);
    assert(input_val >= 0.0);
    assert(input_val <= 1.0);
    float max_val = -input_val < 0 ? 0 : -input_val;

    if (pos_weight_processed == nullptr) {
      out[i] = __float2half((1 - target_val) * input_val + max_val
                            + (logf(expf(-max_val) + expf(-input_val - max_val))));
    } else {
      float pos_weight_processed_val = __half2float(pos_weight_processed[i]) - target_val + 1;
      out[i] = __float2half((1 - target_val) * input_val
                            + (pos_weight_processed_val
                               * (logf(expf(-max_val) + expf(-input_val - max_val)) + max_val)));
    }
    if (weight != nullptr) { out[i] = __hmul(out[i], weight[i]); }
  }
}
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

template<typename T>
__global__ void ComputeBinaryCrossEntropyWithLogitsGradOut(int64_t elem_cnt, const T* input,
                                                           const T* target, const T* dy, T* dx,
                                                           const T* weight,
                                                           const T* pos_weight_processed,
                                                           const ReductionType reduction_type) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T dy_val = reduction_type == ReductionType::kNone ? dy[i] : *dy;
    T input_sigmoid = CalSigmoid(input_val);
    if (pos_weight_processed == nullptr) {
      dx[i] = (input_sigmoid - target_val) * dy_val;
    } else {
      dx[i] =
          dy_val
          * ((pos_weight_processed[i] + 1 - target_val) * input_sigmoid - pos_weight_processed[i]);
    }
    if (weight != nullptr) { dx[i] *= weight[i]; }
    if (reduction_type == ReductionType::kMean) { dx[i] /= elem_cnt; }
  }
}

__global__ void ComputeBinaryCrossEntropyWithLogitsGradOutHalf(int64_t elem_cnt, const half* input,
                                                               const half* target, const half* dy,
                                                               half* dx, const half* weight,
                                                               const half* pos_weight_processed,
                                                               const ReductionType reduction_type) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    float input_val = __half2float(input[i]);
    float target_val = __half2float(target[i]);
    float dy_val = __half2float(reduction_type == ReductionType::kNone ? dy[i] : *dy);
    float input_sigmoid = CalSigmoid(input_val);
    if (pos_weight_processed == nullptr) {
      dx[i] = __float2half((input_sigmoid - target_val) * dy_val);
    } else {
      dx[i] =
          __float2half(dy_val
                       * ((__half2float(pos_weight_processed[i]) + 1 - target_val) * input_sigmoid
                          - __half2float(pos_weight_processed[i])));
    }

    if (weight != nullptr) { dx[i] = __hmul(dx[i], weight[i]); }
    if (reduction_type == ReductionType::kMean) {
      dx[i] = __float2half(__half2float(dx[i]) / elem_cnt);
    }
  }
}

template<typename T>
class BinaryCrossEntropyWithLogitsKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsKernel() = default;
  ~BinaryCrossEntropyWithLogitsKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();
    T* tmp_buffer = tmp_buffer_blob->mut_dptr<T>();
    T* tmp_out = tmp_buffer;

    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    const T* pos_weight = ctx->has_input("pos_weight", 0)
                              ? ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<T>()
                              : nullptr;
    T* pos_weight_processed = nullptr;

    if (ctx->Attr<bool>("has_pos_weight")) {
      if (reduction == ReductionType::kNone) {
        pos_weight_processed = tmp_buffer;
      } else {
        pos_weight_processed = tmp_buffer + elem_cnt;
      }
      Shape pos_weight_shape = Shape::Ones(target_blob->shape().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape().elem_cnt());
      NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
          ctx->device_ctx(), XpuVarNdarray<T>(target_blob->shape(), pos_weight_processed),
          XpuVarNdarray<const T>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const T>(target_blob->shape(), target));
    }
    ComputeBinaryCrossEntropyWithLogitsOut<<<BlocksNum4ThreadsNum(elem_cnt),
                                             kCudaThreadsNumPerBlock, 0,
                                             ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, input, target, reduction == ReductionType::kNone ? out : tmp_out, weight,
        pos_weight_processed);

    if (reduction != ReductionType::kNone) {
      ApplyLossReduction<T>(ctx->device_ctx(), elem_cnt, tmp_out, out, reduction);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<>
class BinaryCrossEntropyWithLogitsKernel<float16> final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsKernel() = default;
  ~BinaryCrossEntropyWithLogitsKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const float16* input = input_blob->dptr<float16>();
    const float16* target = target_blob->dptr<float16>();
    float16* out = out_blob->mut_dptr<float16>();
    float16* tmp_buffer = tmp_buffer_blob->mut_dptr<float16>();
    float16* tmp_out = tmp_buffer;

    const float16* weight = ctx->has_input("weight", 0)
                                ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<float16>()
                                : nullptr;
    const float16* pos_weight = ctx->has_input("pos_weight", 0)
                                    ? ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<float16>()
                                    : nullptr;
    float16* pos_weight_processed = nullptr;
    if (ctx->Attr<bool>("has_pos_weight")) {
      if (reduction == ReductionType::kNone) {
        pos_weight_processed = tmp_buffer;
      } else {
        pos_weight_processed = tmp_buffer + elem_cnt;
      }
      Shape pos_weight_shape = Shape::Ones(target_blob->shape().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape().elem_cnt());
      NdarrayUtil<DeviceType::kGPU, float16>::BroadcastMul(
          ctx->device_ctx(), XpuVarNdarray<float16>(target_blob->shape(), pos_weight_processed),
          XpuVarNdarray<const float16>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const float16>(target_blob->shape(), target));
    }
    ComputeBinaryCrossEntropyWithLogitsOutHalf<<<BlocksNum4ThreadsNum(elem_cnt),
                                                 kCudaThreadsNumPerBlock, 0,
                                                 ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, reinterpret_cast<const half*>(input), reinterpret_cast<const half*>(target),
        reduction == ReductionType::kNone ? reinterpret_cast<half*>(out)
                                          : reinterpret_cast<half*>(tmp_out),
        reinterpret_cast<const half*>(weight), reinterpret_cast<const half*>(pos_weight_processed));

    if (reduction != ReductionType::kNone) {
      ApplyLossReduction<float16>(ctx->device_ctx(), elem_cnt, tmp_out, out, reduction);
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
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    T* tmp_buffer = tmp_buffer_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    const T* pos_weight = ctx->has_input("pos_weight", 0)
                              ? ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<T>()
                              : nullptr;
    T* pos_weight_processed = nullptr;
    if (ctx->Attr<bool>("has_pos_weight")) {
      pos_weight_processed = tmp_buffer;
      Shape pos_weight_shape = Shape::Ones(target_blob->shape().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape().elem_cnt());
      NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
          ctx->device_ctx(), XpuVarNdarray<T>(target_blob->shape(), pos_weight_processed),
          XpuVarNdarray<const T>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const T>(target_blob->shape(), target));
    }
    ComputeBinaryCrossEntropyWithLogitsGradOut<<<BlocksNum4ThreadsNum(elem_cnt),
                                                 kCudaThreadsNumPerBlock, 0,
                                                 ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, input, target, dy, dx, weight, pos_weight_processed, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<>
class BinaryCrossEntropyWithLogitsGradKernel<float16> final : public user_op::OpKernel {
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
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const float16* dy = dy_blob->dptr<float16>();
    const float16* input = input_blob->dptr<float16>();
    const float16* target = target_blob->dptr<float16>();
    float16* dx = dx_blob->mut_dptr<float16>();
    float16* tmp_buffer = tmp_buffer_blob->mut_dptr<float16>();
    const float16* weight = ctx->has_input("weight", 0)
                                ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<float16>()
                                : nullptr;
    const float16* pos_weight = ctx->has_input("pos_weight", 0)
                                    ? ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<float16>()
                                    : nullptr;
    float16* pos_weight_processed = nullptr;
    if (ctx->Attr<bool>("has_pos_weight")) {
      pos_weight_processed = tmp_buffer;
      Shape pos_weight_shape = Shape::Ones(target_blob->shape().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape().elem_cnt());
      NdarrayUtil<DeviceType::kGPU, float16>::BroadcastMul(
          ctx->device_ctx(), XpuVarNdarray<float16>(target_blob->shape(), pos_weight_processed),
          XpuVarNdarray<const float16>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const float16>(target_blob->shape(), target));
    }
    ComputeBinaryCrossEntropyWithLogitsGradOutHalf<<<BlocksNum4ThreadsNum(elem_cnt),
                                                     kCudaThreadsNumPerBlock, 0,
                                                     ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, reinterpret_cast<const half*>(input), reinterpret_cast<const half*>(target),
        reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx),
        reinterpret_cast<const half*>(weight), reinterpret_cast<const half*>(pos_weight_processed),
        reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape("input", 0).elem_cnt();
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));
    size_t tmp_buffer_size = 0;
    if (ctx->Attr<bool>("has_pos_weight")) { tmp_buffer_size += GetCudaAlignedSize(n * sizeof(T)); }
    if (reduction != ReductionType::kNone) { tmp_buffer_size += GetCudaAlignedSize(n * sizeof(T)); }
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

#define REGISTER_BINARY_CROSS_ENTROPY_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits")                                \
      .SetCreateFn<BinaryCrossEntropyWithLogitsKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                      \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<dtype>());

#define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(dtype)                                  \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_grad")                           \
      .SetCreateFn<BinaryCrossEntropyWithLogitsGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                      \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn(GenBwInferTmpSizeFn<dtype>());

REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(double)
REGISTER_BINARY_CROSS_ENTROPY_KERNEL(float16)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double)
REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float16)

}  // namespace user_op
}  // namespace oneflow
