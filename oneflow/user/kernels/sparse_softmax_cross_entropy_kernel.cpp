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
#include "oneflow/user/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/user/kernels/softmax_kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyKernel() = default;
  ~SparseSoftmaxCrossEntropyKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_instances = label->shape().elem_cnt();
    CHECK_EQ(prediction->shape().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prediction->shape().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");
    SoftmaxKernelUtil<device_type, T>::ComputeProb(
        ctx->device_ctx(), num_instances, num_classes, prediction->dptr<T>(), out->mut_dptr<T>(),
        prob->mut_dptr<T>(), tmp_buffer->mut_dptr(), tmp_buffer->shape().elem_cnt() * sizeof(T));
    SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeEntropy(
        ctx->device_ctx(), num_instances, num_classes, depth, lower_bound, prob->dptr<T>(),
        label->dptr<K>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyMsKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyMsKernel() = default;
  ~SparseSoftmaxCrossEntropyMsKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(FATAL) << "SparseSoftmaxCrossEntropyMsKernel should be split to ops";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL(kernel_class, kernel_name, device_type_v, \
                                                     dtype_pair, ltype_pair)                   \
  REGISTER_USER_KERNEL(kernel_name)                                                            \
      .SetCreateFn<kernel_class<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                   \
                                OF_PP_PAIR_FIRST(ltype_pair)>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type_v)                              \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair))   \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                      \
        const Shape* prediction_shape = ctx->Shape4ArgNameAndIndex("prediction", 0);           \
        return prediction_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair));            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyKernel),
                                 ("sparse_softmax_cross_entropy"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyKernel),
                                 ("sparse_softmax_cross_entropy"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsKernel),
                                 ("sparse_softmax_cross_entropy_ms"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsKernel),
                                 ("sparse_softmax_cross_entropy_ms"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyGradKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyGradKernel() = default;
  ~SparseSoftmaxCrossEntropyGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* prediction_diff = ctx->Tensor4ArgNameAndIndex("prediction_diff", 0);
    const int64_t num_instances = label->shape().elem_cnt();
    CHECK_EQ(prob->shape().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prob->shape().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");
    SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeDiffWithSoftmax(
        ctx->device_ctx(), prediction_diff->shape().elem_cnt(), num_classes, depth, lower_bound,
        prob->dptr<T>(), label->dptr<K>(), dy->dptr<T>(), prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyMsGradKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyMsGradKernel() = default;
  ~SparseSoftmaxCrossEntropyMsGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* prediction_diff = ctx->Tensor4ArgNameAndIndex("prediction_diff", 0);
    const int64_t num_instances = label->shape().elem_cnt();
    CHECK_EQ(prob->shape().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prob->shape().elem_cnt() / num_instances;
    const int64_t depth = ctx->Attr<int64_t>("depth");
    int64_t lower_bound = 0;
    if (ctx->parallel_ctx().parallel_num() > 1) {
      BalancedSplitter bs(depth, ctx->parallel_ctx().parallel_num());
      lower_bound = bs.At(ctx->parallel_ctx().parallel_id()).begin();
    }
    SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeDiffWithSoftmax(
        ctx->device_ctx(), prediction_diff->shape().elem_cnt(), num_classes, depth, lower_bound,
        prob->dptr<T>(), label->dptr<K>(), dy->dptr<T>(), prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL(kernel_class, kernel_name,             \
                                                          device_type_v, dtype_pair, ltype_pair) \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn<kernel_class<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                     \
                                OF_PP_PAIR_FIRST(ltype_pair)>>()                                 \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == device_type_v)                                             \
          & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair))                  \
          & (user_op::HobDataType("prediction_diff", 0) == OF_PP_PAIR_SECOND(dtype_pair)))       \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                     \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {  \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("prediction_diff", 0, "prob", 0, true));          \
        return Maybe<void>::Ok();                                                                \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyGradKernel),
                                 ("sparse_softmax_cross_entropy_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyGradKernel),
                                 ("sparse_softmax_cross_entropy_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsGradKernel),
                                 ("sparse_softmax_cross_entropy_ms_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsGradKernel),
                                 ("sparse_softmax_cross_entropy_ms_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif
}  // namespace user_op
}  // namespace oneflow
