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
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/user/kernels/nll_prob_kernel_util.h"

namespace oneflow {

namespace {

class NLLProbKernelCache final : public user_op::OpKernelCache {
 public:
  explicit NLLProbKernelCache(int64_t num_classes) : num_classes_(num_classes) {}
  ~NLLProbKernelCache() override = default;

  int64_t num_classes() const { return num_classes_; }

 private:
  const int64_t num_classes_;
};

std::shared_ptr<user_op::OpKernelCache> CreateNLLProbKernelCache(user_op::KernelCacheContext* ctx) {
  CHECK_GT(ctx->parallel_ctx().parallel_num(), 0) << ctx->op_name() << ": invalid parallel_ctx";
  if (ctx->parallel_ctx().parallel_num() == 1) { return nullptr; }

  const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("input", 0);
  const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
  CHECK_EQ(nd_sbp.sbp_parallel_size(), hierarchy.NumAxes())
      << ctx->op_name() << ": Expected input sbp " << NdSbpToString(nd_sbp) << " match hierarchy "
      << hierarchy.ToString();

  const Shape& shape = ctx->LogicalTensorDesc4ArgNameAndIndex("input", 0)->shape();
  const int64_t class_axis = shape.NumAxes() - 1;

  bool split_class_dim = false;
  for (const auto& sbp : nd_sbp.sbp_parallel()) {
    if (sbp.has_split_parallel() && sbp.split_parallel().axis() == class_axis) {
      split_class_dim = true;
      break;
    }
  }

  if (!split_class_dim) { return nullptr; }

  TensorSliceView view =
      GetTensorSliceView4ParallelId(hierarchy, nd_sbp, shape, ctx->parallel_ctx().parallel_id());
  return std::make_shared<NLLProbKernelCache>(view.At(class_axis).size());
}

}  // namespace

template<DeviceType device_type, typename T>
class NLLProbKernel final : public user_op::OpKernel {
 public:
  NLLProbKernel() = default;
  ~NLLProbKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateNLLProbKernelCache(ctx);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* output = ctx->Tensor4ArgNameAndIndex("output", 0);

    // const int64_t N = target->shape_view().elem_cnt();
    const int64_t N = input->shape_view().At(0);
    const int64_t C = input->shape_view().At(input->shape_view().NumAxes() - 1);
    CHECK_LE(N, std::numeric_limits<int32_t>::max())
        << "Expected batch size not exceed int32 numeric limits";

    if (cache) {
      const auto* spec_cache = dynamic_cast<const NLLProbKernelCache*>(cache);
      CHECK_NOTNULL(spec_cache);
      CHECK_EQ(spec_cache->num_classes(), C) << ctx->op_name() << ": expected num_classes " << C
                                             << ", got " << spec_cache->num_classes();
    }

    const double label_smoothing = ctx->Attr<double>("label_smoothing");
    const double one_minus_label_smoothing = static_cast<double>(1) - label_smoothing;
    const double label_smoothing_rest_factor = label_smoothing / static_cast<double>(C);

    const T* weight_dptr = nullptr;
    if (ctx->has_input("weight", 0)) {
      weight_dptr = CHECK_NOTNULL(ctx->Tensor4ArgNameAndIndex("weight", 0))->dptr<T>();
    }
    if (label_smoothing > 0) {
      NLLProbKernelUtil<device_type, T, true>::Forward(
          ctx->stream(), static_cast<int32_t>(N), static_cast<int64_t>(C), input->dptr<T>(),
          target->dptr<T>(), weight_dptr, static_cast<T>(one_minus_label_smoothing),
          static_cast<T>(label_smoothing_rest_factor), output->mut_dptr<T>());
    } else {
      NLLProbKernelUtil<device_type, T, false>::Forward(
          ctx->stream(), static_cast<int32_t>(N), static_cast<int64_t>(C), input->dptr<T>(),
          target->dptr<T>(), weight_dptr, static_cast<T>(one_minus_label_smoothing),
          static_cast<T>(label_smoothing_rest_factor), output->mut_dptr<T>());
    }
  }
};

template<DeviceType device_type, typename T>
class NLLProbGradKernel final : public user_op::OpKernel {
 public:
  NLLProbGradKernel() = default;
  ~NLLProbGradKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateNLLProbKernelCache(ctx);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* target = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* out_grad = ctx->Tensor4ArgNameAndIndex("out_grad", 0);
    auto* in_grad = ctx->Tensor4ArgNameAndIndex("in_grad", 0);

    const int64_t N = in_grad->shape_view().At(0);  // batch size
    const int64_t C = in_grad->shape_view().At(in_grad->shape_view().NumAxes() - 1);
    CHECK_LE(N, std::numeric_limits<int32_t>::max())
        << "Expected batch size not exceed int32 numeric limits";

    if (cache) {
      const auto* spec_cache = dynamic_cast<const NLLProbKernelCache*>(cache);
      CHECK_NOTNULL(spec_cache);
      CHECK_EQ(spec_cache->num_classes(), C) << ctx->op_name() << ": expected num_classes " << C
                                             << ", got " << spec_cache->num_classes();
    }

    const T* weight_dptr = nullptr;
    if (ctx->has_input("weight", 0)) {
      weight_dptr = CHECK_NOTNULL(ctx->Tensor4ArgNameAndIndex("weight", 0))->dptr<T>();
    }
    const double label_smoothing = ctx->Attr<double>("label_smoothing");
    const double one_minus_label_smoothing = static_cast<double>(1) - label_smoothing;
    const double label_smoothing_rest_factor = label_smoothing / static_cast<double>(C);
    if (label_smoothing > 0) {
      NLLProbKernelUtil<device_type, T, true>::Backward(
          ctx->stream(), static_cast<int32_t>(N), static_cast<int64_t>(C), out_grad->dptr<T>(),
          target->dptr<T>(), weight_dptr, static_cast<T>(one_minus_label_smoothing),
          static_cast<T>(label_smoothing_rest_factor), in_grad->mut_dptr<T>());
    } else {
      NLLProbKernelUtil<device_type, T, false>::Backward(
          ctx->stream(), static_cast<int32_t>(N), static_cast<int64_t>(C), out_grad->dptr<T>(),
          target->dptr<T>(), weight_dptr, static_cast<T>(one_minus_label_smoothing),
          static_cast<T>(label_smoothing_rest_factor), in_grad->mut_dptr<T>());
    }
  };
};

#define REGISTER_NLL_PROB_KERNELS(device, dtype)                                             \
  REGISTER_USER_KERNEL("nll_prob")                                                           \
      .SetCreateFn<NLLProbKernel<device, dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                  \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)    \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("nll_prob_grad")                                                      \
      .SetCreateFn<NLLProbGradKernel<device, dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                  \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)    \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("out_grad", 0) == GetDataType<dtype>::value))

REGISTER_NLL_PROB_KERNELS(DeviceType::kCPU, float);
REGISTER_NLL_PROB_KERNELS(DeviceType::kCPU, double);

#ifdef WITH_CUDA

REGISTER_NLL_PROB_KERNELS(DeviceType::kCUDA, float);
REGISTER_NLL_PROB_KERNELS(DeviceType::kCUDA, double);
REGISTER_NLL_PROB_KERNELS(DeviceType::kCUDA, half);

#endif  // WITH_CUDA

}  // namespace oneflow
