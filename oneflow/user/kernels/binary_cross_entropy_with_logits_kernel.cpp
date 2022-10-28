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
inline T ComputeMaxVal(const T x) {
  T y = -x;
  return y < 0 ? 0 : y;
}

template<typename T>
inline T CalSigmoid(const T x) {
  const T half_of_one = static_cast<T>(0.5);
  return half_of_one * std::tanh(half_of_one * x) + half_of_one;
}

template<typename T>
void ComputeBinaryCrossEntropyWithLogitsOut(int64_t elem_cnt, const T* input, const T* target,
                                            T* out, const T* weight,
                                            const T* pos_weight_processed) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T max_val = ComputeMaxVal(input_val);
    if (out != nullptr) {
      if (pos_weight_processed == nullptr) {
        out[i] = (1 - target_val) * input_val + max_val
                 + (std::log(std::exp(-max_val) + std::exp(-input_val - max_val)));
      } else {
        T pos_weight_processed_val = pos_weight_processed[i] - target_val + 1;
        out[i] = (1 - target_val) * input_val
                 + (pos_weight_processed_val
                    * (std::log(std::exp(-max_val) + std::exp(-input_val - max_val)) + max_val));
      }
    }
    if (weight != nullptr && out != nullptr) { out[i] *= weight[i]; }
  }
}
template<typename T>
void ComputeBinaryCrossEntropyWithLogitsGradOut(int64_t elem_cnt, const T* input, const T* target,
                                                const T* dy, T* dx, const T* weight,
                                                const T* pos_weight_processed) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T dy_val = dy[i];
    T input_sigmoid = CalSigmoid(input_val);
    if (pos_weight_processed == nullptr) {
      dx[i] = (input_sigmoid - target_val) * dy_val;
    } else {
      dx[i] =
          dy_val
          * ((pos_weight_processed[i] + 1 - target_val) * input_sigmoid - pos_weight_processed[i]);
    }

    if (weight != nullptr) { dx[i] *= weight[i]; }
  }
}

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

    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;

    T* pos_weight_processed = nullptr;

    if (ctx->Attr<bool>("has_pos_weight")) {
      pos_weight_processed = tmp_buffer_blob->mut_dptr<T>();
      const T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      NdarrayUtil<DeviceType::kCPU, T>::BroadcastMul(
          ctx->stream(), XpuVarNdarray<T>(target_blob->shape_view(), pos_weight_processed),
          XpuVarNdarray<const T>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const T>(target_blob->shape_view(), target));
    }
    ComputeBinaryCrossEntropyWithLogitsOut(elem_cnt, input, target, out, weight,
                                           pos_weight_processed);
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
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;

    T* pos_weight_processed = nullptr;

    if (ctx->Attr<bool>("has_pos_weight")) {
      pos_weight_processed = tmp_buffer_blob->mut_dptr<T>();
      const T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      NdarrayUtil<DeviceType::kCPU, T>::BroadcastMul(
          ctx->stream(), XpuVarNdarray<T>(target_blob->shape_view(), pos_weight_processed),
          XpuVarNdarray<const T>(pos_weight_shape, pos_weight),
          XpuVarNdarray<const T>(target_blob->shape_view(), target));
    }
    ComputeBinaryCrossEntropyWithLogitsGradOut(elem_cnt, input, target, dy, dx, weight,
                                               pos_weight_processed);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape("target", 0).elem_cnt();
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

#define REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits")                                 \
      .SetCreateFn<BinaryCrossEntropyWithLogitsKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<dtype>());

#define REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_grad")                            \
      .SetCreateFn<BinaryCrossEntropyWithLogitsGradKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn(GenBwInferTmpSizeFn<dtype>());

REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(double)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
