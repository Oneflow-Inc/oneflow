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
#include "oneflow/user/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/cuda/softmax.cuh"

namespace oneflow {
namespace user_op {

namespace {

template<typename T>
void ComputeProb(DeviceCtx* ctx, const int64_t row, const int64_t col, const T* in, T* prob) {
  cuda::softmax::DirectFetch<T> fetch(in, col);
  cuda::softmax::DirectStore<T> store(prob, col);
  cuda::softmax::DispatchSoftmax<decltype(fetch), decltype(store), T>(ctx->cuda_stream(), fetch,
                                                                      store, row, col);
}

template<>
void ComputeProb(DeviceCtx* ctx, const int64_t row, const int64_t col, const float16* in,
                 float16* prob) {
  cuda::softmax::DirectFetch<half> fetch(reinterpret_cast<const half*>(in), col);
  cuda::softmax::DirectStore<half> store(reinterpret_cast<half*>(prob), col);
  cuda::softmax::DispatchSoftmax<decltype(fetch), decltype(store), half>(ctx->cuda_stream(), fetch,
                                                                         store, row, col);
}

}  // namespace

template<typename T, typename K>
class SparseSoftmaxCrossEntropyKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyKernel() = default;
  ~SparseSoftmaxCrossEntropyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_instances = label->shape().elem_cnt();
    CHECK_EQ(prediction->shape().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prediction->shape().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");
    ComputeProb(ctx->device_ctx(), num_instances, num_classes, prediction->dptr<T>(),
                prob->mut_dptr<T>());
    SparseCrossEntropyKernelUtil<DeviceType::kGPU, T, K>::ComputeEntropy(
        ctx->device_ctx(), num_instances, num_classes, depth, lower_bound, prob->dptr<T>(),
        label->dptr<K>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL(dtype_pair, ltype_pair)                 \
  REGISTER_USER_KERNEL("sparse_softmax_cross_entropy")                                       \
      .SetCreateFn<SparseSoftmaxCrossEntropyKernel<OF_PP_PAIR_FIRST(dtype_pair),             \
                                                   OF_PP_PAIR_FIRST(ltype_pair)>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                         \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair)) \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
