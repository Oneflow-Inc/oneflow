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
#include "oneflow/user/kernels/binary_cross_entropy_with_logits_mean_kernel_util.h"
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
void ComputeBinaryCrossEntropyWithLogitsReduceMeanOut(int64_t elem_cnt, const T* input,
                                                      const T* target, T* out,
                                                      int64_t reduce_elem_cnt) {
  T result = 0.0;
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T max_val = ComputeMaxVal(input_val);
    result += (1 - target_val) * input_val + max_val
              + (std::log(std::exp(-max_val) + std::exp(-input_val - max_val)));
  }
  out[0] = result / reduce_elem_cnt;
}

template<typename T>
void ComputeBinaryCrossEntropyWithLogitsReduceMeanGradOut(int64_t elem_cnt, const T* input,
                                                          const T* target, const T* dy, T* dx,
                                                          int64_t reduce_elem_cnt) {
  T dy_val = dy[0] / reduce_elem_cnt;
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    T input_val = input[i];
    T target_val = target[i];
    T input_sigmoid = CalSigmoid(input_val);
    dx[i] = (input_sigmoid - target_val) * dy_val;
  }
}

template<typename T>
class BinaryCrossEntropyWithLogitsReduceMeanKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsReduceMeanKernel() = default;
  ~BinaryCrossEntropyWithLogitsReduceMeanKernel() = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateBCEWithLogitsReduceMeanKernelCache(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t local_elem_cnt = input_blob->shape_view().elem_cnt();
    int64_t reduce_elem_cnt = local_elem_cnt;
    if (cache != nullptr) {
      // Because `out`'s SBP maybe P or B, we need to use reduce_elem_cnt as reduce_mean factor.
      const auto* bce_cache = dynamic_cast<const BCEWithLogitsReduceMeanKernelCache*>(cache);
      CHECK_NOTNULL(bce_cache);
      reduce_elem_cnt = bce_cache->reduce_elem_cnt();
    }

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();

    ComputeBinaryCrossEntropyWithLogitsReduceMeanOut(local_elem_cnt, input, target, out,
                                                     reduce_elem_cnt);
  }
};

template<typename T>
class BinaryCrossEntropyWithLogitsReduceMeanGradKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsReduceMeanGradKernel() = default;
  ~BinaryCrossEntropyWithLogitsReduceMeanGradKernel() = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateBCEWithLogitsReduceMeanKernelCache(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);

    int64_t local_elem_cnt = input_blob->shape_view().elem_cnt();
    int64_t reduce_elem_cnt = local_elem_cnt;
    if (cache != nullptr) {
      // Because `out`'s SBP maybe P or B, we need to use reduce_elem_cnt as reduce_mean factor.
      const auto* bce_cache = dynamic_cast<const BCEWithLogitsReduceMeanKernelCache*>(cache);
      CHECK_NOTNULL(bce_cache);
      reduce_elem_cnt = bce_cache->reduce_elem_cnt();
    }

    const T* dy = dy_blob->dptr<T>();
    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    ComputeBinaryCrossEntropyWithLogitsReduceMeanGradOut(local_elem_cnt, input, target, dy, dx,
                                                         reduce_elem_cnt);
  }
};

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_REDUCE_MEAN_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_reduce_mean")                     \
      .SetCreateFn<BinaryCrossEntropyWithLogitsReduceMeanKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_REDUCE_MEAN_GRAD_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_reduce_mean_grad")                \
      .SetCreateFn<BinaryCrossEntropyWithLogitsReduceMeanGradKernel<dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_REDUCE_MEAN_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_REDUCE_MEAN_KERNEL(double)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_REDUCE_MEAN_GRAD_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_REDUCE_MEAN_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
