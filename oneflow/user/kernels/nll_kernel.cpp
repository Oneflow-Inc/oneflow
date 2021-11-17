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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T, typename K>
void ComputeNllOut(int64_t num_instances, K num_classes, K ignore_index, const T* input,
                   const K* target, T* out, const T* weight, T* total_weight) {
  *total_weight = 0;
  FOR_RANGE(int64_t, i, 0, num_instances) {
    K label = target[i];
    if (label == ignore_index) {
      out[i] = 0;
      continue;
    }
    CHECK_GE(label, 0);
    CHECK_LT(label, num_classes);
    T cur_weight = weight == nullptr ? 1 : weight[label];
    *total_weight += cur_weight;
    out[i] = -input[i * num_classes + label] * cur_weight;
  }
}
template<typename T, typename K>
void ComputeNllGradOut(int64_t num_instances, K num_classes, K ignore_index, const K* target,
                       const T* dy, T* dx, const T* weight, const T* total_weight,
                       const ReductionType reduction_type) {
  FOR_RANGE(int64_t, i, 0, num_instances) {
    K label = target[i];
    if (label == ignore_index) { continue; }
    CHECK_GE(label, 0);
    CHECK_LT(label, num_classes);
    T cur_weight = weight == nullptr ? -1 : -weight[label];
    dx[i * num_classes + label] =
        (reduction_type == ReductionType::kNone ? dy[i] : (*dy)) * cur_weight;
    if (reduction_type == ReductionType::kMean) { dx[i * num_classes + label] /= *total_weight; }
  }
}
template<typename T, typename K>
class NllKernel final : public user_op::OpKernel {
 public:
  NllKernel() = default;
  ~NllKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);

    const int64_t num_instances = target_blob->shape().elem_cnt();
    CHECK_EQ(input_blob->shape().elem_cnt() % num_instances, 0);
    const K num_classes = static_cast<K>(input_blob->shape().elem_cnt() / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const T* input = input_blob->dptr<T>();
    const K* target = target_blob->dptr<K>();
    T* out = out_blob->mut_dptr<T>();
    T* total_weight = total_weight_blob->mut_dptr<T>();
    T* tmp_out = reduction == ReductionType::kNone ? nullptr : tmp_buffer_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;

    ComputeNllOut(num_instances, num_classes, ignore_index, input, target,
                  reduction == ReductionType::kNone ? out : tmp_out, weight, total_weight);
    if (reduction == ReductionType::kNone) return;

    *out = 0;
    FOR_RANGE(int64_t, i, 0, num_instances) { *out += tmp_out[i]; }
    if (reduction == ReductionType::kSum) return;

    *out /= *total_weight;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename K>
class NllGradKernel final : public user_op::OpKernel {
 public:
  NllGradKernel() = default;
  ~NllGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);

    const int64_t num_instances = target_blob->shape().elem_cnt();
    const int64_t input_elem_cnt = input_blob->shape().elem_cnt();
    CHECK_EQ(input_elem_cnt % num_instances, 0);
    const K num_classes = static_cast<K>(input_elem_cnt / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const T* dy = dy_blob->dptr<T>();
    const K* target = target_blob->dptr<K>();
    const T* total_weight = total_weight_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    Memset<DeviceType::kCPU>(ctx->device_ctx(), dx, 0,
                             GetCudaAlignedSize(input_elem_cnt * sizeof(T)));
    ComputeNllGradOut(num_instances, num_classes, ignore_index, target, dy, dx, weight,
                      total_weight, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
#define REGISTER_NLL_KERNEL(dtype_pair, ltype_pair)                                           \
  REGISTER_USER_KERNEL("nll")                                                                 \
      .SetCreateFn<NllKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>()   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                         \
                       & (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair)) \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))   \
      .SetInferTmpSizeFn(loss::GenDefaultInferTmpSizeFn<OF_PP_PAIR_FIRST(dtype_pair)>());

#define REGISTER_NLL_GRAD_KERNEL(dtype_pair, ltype_pair)                                        \
  REGISTER_USER_KERNEL("nll_grad")                                                              \
      .SetCreateFn<NllGradKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       & (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair))   \
                       & (user_op::HobDataType("dy", 0) == OF_PP_PAIR_SECOND(dtype_pair))       \
                       & (user_op::HobDataType("dx", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_KERNEL, FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_GRAD_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
}  // namespace user_op
}  // namespace oneflow
