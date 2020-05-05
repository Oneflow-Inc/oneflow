#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/customized/kernels/softmax_kernel_util.h"

namespace oneflow {

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
    SoftmaxKernelUtil<device_type, T>::ComputeProb(
        ctx->device_ctx(), num_instances, num_classes, prediction->dptr<T>(), out->mut_dptr<T>(),
        prob->mut_dptr<T>(), tmp_buffer->mut_dptr(), tmp_buffer->shape().elem_cnt() * sizeof(T));
    SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeEntropy(
        ctx->device_ctx(), num_instances, num_classes, prob->dptr<T>(), label->dptr<K>(),
        out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL(device_type_v, dtype_pair, ltype_pair)     \
  REGISTER_USER_KERNEL("sparse_softmax_cross_entropy")                                          \
      .SetCreateFn<SparseSoftmaxCrossEntropyKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                                   OF_PP_PAIR_FIRST(ltype_pair)>>()             \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* label_desc = ctx.TensorDesc4ArgNameAndIndex("label", 0);     \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        return ctx.device_type() == device_type_v                                               \
               && label_desc->data_type() == OF_PP_PAIR_SECOND(ltype_pair)                      \
               && out_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                       \
      })                                                                                        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                       \
        const Shape* prediction_shape = ctx->Shape4ArgNameAndIndex("prediction", 0);            \
        return prediction_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair));             \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

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

    Memcpy<device_type>(ctx->device_ctx(), prediction_diff->mut_dptr<T>(), prob->dptr<T>(),
                        prediction_diff->shape().elem_cnt() * sizeof(T));
    SparseCrossEntropyKernelUtil<device_type, T, K>::BackwardSub(
        ctx->device_ctx(), num_instances, num_classes, label->dptr<K>(), dy->dptr<T>(),
        prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL(device_type_v, dtype_pair, ltype_pair) \
  REGISTER_USER_KERNEL("sparse_softmax_cross_entropy_grad")                                      \
      .SetCreateFn<SparseSoftmaxCrossEntropyGradKernel<                                          \
          device_type_v, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>()          \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* label_desc = ctx.TensorDesc4ArgNameAndIndex("label", 0);      \
        const user_op::TensorDesc* prediction_diff_desc =                                        \
            ctx.TensorDesc4ArgNameAndIndex("prediction_diff", 0);                                \
        return ctx.device_type() == device_type_v                                                \
               && label_desc->data_type() == OF_PP_PAIR_SECOND(ltype_pair)                       \
               && prediction_diff_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);            \
      })                                                                                         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                     \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {  \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("prediction_diff", 0, "prob", 0, true));          \
        return Maybe<void>::Ok();                                                                \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
