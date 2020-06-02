#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
bool IsKernelMatched(const user_op::KernelRegContext& ctx) {
  const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);
  return ctx.device_type() == device_type && y_desc->data_type() == GetDataType<T>::value;
}

size_t ReduceSumLikeInferTmpSize(user_op::InferContext* ctx) {
  const auto& reduce_axes_vec = ctx->Attr<AxisVector>("reduce_axes");
  if (reduce_axes_vec.empty()) { return 0; }
  const user_op::TensorDesc* tensor_desc_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  return tensor_desc_x->shape().elem_cnt() * GetSizeOfDataType(tensor_desc_x->data_type());
}

template<DeviceType device_type, typename T>
class ReduceSumLikeOpKernel final : public user_op::OpKernel {
 public:
  ReduceSumLikeOpKernel() = default;
  ~ReduceSumLikeOpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& reduce_axes_vec = ctx->Attr<AxisVector>("reduce_axes");
    if (reduce_axes_vec.empty()) {
      CHECK_EQ(tensor_x->shape(), tensor_y->shape());
      Memcpy<device_type>(ctx->device_ctx(), tensor_y->mut_dptr(), tensor_x->dptr(),
                          tensor_x->shape().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    } else {
      user_op::Tensor* tensor_tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      T* temp_storage = tensor_tmp->mut_dptr<T>();
      int64_t num_axes = tensor_x->shape().NumAxes();
      Shape y_extend_shape = CreateLeftExtendedShape(tensor_y->shape(), num_axes);
      NdarrayUtil<device_type, T>::ReduceSum(
          ctx->device_ctx(), XpuVarNdarray<T>(y_extend_shape, tensor_y->mut_dptr<T>()),
          XpuVarNdarray<const T>(tensor_x->shape(), tensor_x->dptr<T>()),
          XpuVarNdarray<T>(tensor_x->shape(), temp_storage));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_REDUCE_SUM_LIKE_KERNEL(device, dtype_pair)                       \
  REGISTER_USER_KERNEL("reduce_sum_like")                                         \
      .SetCreateFn<ReduceSumLikeOpKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedPred(IsKernelMatched<device, OF_PP_PAIR_FIRST(dtype_pair)>)    \
      .SetInferTmpSizeFn(ReduceSumLikeInferTmpSize);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_SUM_LIKE_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
