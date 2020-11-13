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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/math_unary_elementwise_func.h"

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOffset(const int64_t batch_idx, const int64_t num_classes,
                             const int64_t lower_bound, const K* label) {
  const int64_t idx = label[batch_idx] - lower_bound;
  if (idx >= 0 && idx < num_classes) {
    return batch_idx * num_classes + idx;
  } else {
    return -1;
  }
}

template<typename T, typename K>
__global__ void GpuForward(const int64_t num_instances, const int64_t num_classes,
                           const int64_t lower_bound, const T m1, const T m2, const T m3,
                           const T* in, const K* label, T* out, T* theta) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int64_t idx = GetOffset<K>(i, num_classes, lower_bound, label);
    T theta_val = 0;
    if (idx != -1) {
      const T cos_theta = in[idx];
      theta_val = AcosFunctor<T>::Forward(cos_theta);
      out[idx] = CosFunctor<T>::Forward(theta_val * m1 + m2) - m3;
    }
    theta[i] = theta_val;
  }
}

template<typename T, typename K>
__global__ void GpuBackward(const int64_t num_instances, const int64_t num_classes,
                            const int64_t lower_bound, const T m1, const T m2, const T m3,
                            const T* dy, const K* label, const T* theta, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int64_t idx = GetOffset<K>(i, num_classes, lower_bound, label);
    if (idx != -1) {
      const T theta_val = theta[i];
      dx[idx] = dy[idx] * SinFunctor<T>::Forward(theta_val * m1 + m2) * m1
                / SinFunctor<T>::Forward(theta_val);
    }
  }
}

class CombinedMarginLossOpKernelState final : public user_op::OpKernelState {
 public:
  CombinedMarginLossOpKernelState(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~CombinedMarginLossOpKernelState() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

std::shared_ptr<user_op::OpKernelState> CreateCombinedMarginLossOpKernelState(
    user_op::KernelInitContext* ctx, const std::string& in_arg_name) {
  const SbpParallel& in_sbp = ctx->SbpParallel4ArgNameAndIndex(in_arg_name, 0);
  if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == 1
      && ctx->parallel_ctx().parallel_num() > 1) {
    CHECK(ctx->SbpParallel4ArgNameAndIndex("label", 0).has_broadcast_parallel());
    const user_op::TensorDesc* in_logical_desc =
        ctx->LogicalTensorDesc4ArgNameAndIndex(in_arg_name, 0);
    BalancedSplitter bs(ctx->Attr<int64_t>("depth"), ctx->parallel_ctx().parallel_num());
    return std::make_shared<CombinedMarginLossOpKernelState>(
        bs.At(ctx->parallel_ctx().parallel_id()).begin(),
        bs.At(ctx->parallel_ctx().parallel_id()).end());
  } else {
    return std::shared_ptr<user_op::OpKernelState>(nullptr);
  }
}

}  // namespace

template<typename T, typename K>
class CombinedMarginLossGpuKernel final : public user_op::OpKernel {
 public:
  CombinedMarginLossGpuKernel() = default;
  ~CombinedMarginLossGpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateCombinedMarginLossOpKernelState(ctx, "x");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* theta = ctx->Tensor4ArgNameAndIndex("theta", 0);
    const float m1 = ctx->Attr<float>("m1");
    const float m2 = ctx->Attr<float>("m2");
    const float m3 = ctx->Attr<float>("m3");
    int64_t lower_bound = 0;
    if (state != nullptr) {
      auto* kernel_state = dynamic_cast<CombinedMarginLossOpKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      CHECK_EQ(x->shape().Count(1), kernel_state->upper() - kernel_state->lower());
      lower_bound = kernel_state->lower();
    }
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), y->mut_dptr<void>(), x->dptr<void>(),
                             x->shape().elem_cnt() * GetSizeOfDataType(x->data_type()));
    GpuForward<<<BlocksNum4ThreadsNum(x->shape().At(0)), kCudaThreadsNumPerBlock, 0,
                 ctx->device_ctx()->cuda_stream()>>>(
        x->shape().At(0), x->shape().Count(1), lower_bound, static_cast<T>(m1), static_cast<T>(m2),
        static_cast<T>(m3), x->dptr<T>(), label->dptr<K>(), y->mut_dptr<T>(), theta->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COMBINED_MARGIN_LOSS_KERNEL(in_type, indices_type)                   \
  REGISTER_USER_KERNEL("combined_margin_loss")                                        \
      .SetCreateFn<CombinedMarginLossGpuKernel<OF_PP_PAIR_FIRST(in_type),             \
                                               OF_PP_PAIR_FIRST(indices_type)>>()     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(in_type)) \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_COMBINED_MARGIN_LOSS_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

template<typename T, typename K>
class CombinedMarginLossGradGpuKernel final : public user_op::OpKernel {
 public:
  CombinedMarginLossGradGpuKernel() = default;
  ~CombinedMarginLossGradGpuKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateCombinedMarginLossOpKernelState(ctx, "dy");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* theta = ctx->Tensor4ArgNameAndIndex("theta", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float m1 = ctx->Attr<float>("m1");
    const float m2 = ctx->Attr<float>("m2");
    const float m3 = ctx->Attr<float>("m3");
    int64_t lower_bound = 0;
    if (state != nullptr) {
      auto* kernel_state = dynamic_cast<CombinedMarginLossOpKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      CHECK_EQ(dy->shape().Count(1), kernel_state->upper() - kernel_state->lower());
      lower_bound = kernel_state->lower();
    }
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), dx->mut_dptr<void>(), dy->dptr<void>(),
                             dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
    GpuBackward<<<BlocksNum4ThreadsNum(dy->shape().At(0)), kCudaThreadsNumPerBlock, 0,
                  ctx->device_ctx()->cuda_stream()>>>(
        dy->shape().At(0), dy->shape().Count(1), lower_bound, static_cast<T>(m1),
        static_cast<T>(m2), static_cast<T>(m3), dy->dptr<T>(), label->dptr<K>(), theta->dptr<T>(),
        dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COMBINED_MARGIN_LOSS_GRAD_KERNEL(dy_type, indices_type)               \
  REGISTER_USER_KERNEL("combined_margin_loss_grad")                                    \
      .SetCreateFn<CombinedMarginLossGradGpuKernel<OF_PP_PAIR_FIRST(dy_type),          \
                                                   OF_PP_PAIR_FIRST(indices_type)>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dy", 0) == OF_PP_PAIR_SECOND(dy_type)) \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_COMBINED_MARGIN_LOSS_GRAD_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
