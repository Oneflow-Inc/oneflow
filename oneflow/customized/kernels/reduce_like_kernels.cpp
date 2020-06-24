#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

size_t ReduceSumLikeInferTmpSize(user_op::InferContext* ctx) {
  if (ctx->Attr<std::vector<int32_t>>("axis").empty()) { return 0; }
  const user_op::TensorDesc* tensor_desc_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  return tensor_desc_x->shape().elem_cnt() * GetSizeOfDataType(tensor_desc_x->data_type());
}

}  // namespace

template<DeviceType device_type, typename T>
class ReduceSumLikeOpKernel final : public user_op::OpKernel {
 public:
  ReduceSumLikeOpKernel() = default;
  ~ReduceSumLikeOpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    if (axis.empty()) {
      CHECK_EQ(tensor_x->shape(), tensor_y->shape());
      Memcpy<device_type>(ctx->device_ctx(), tensor_y->mut_dptr(), tensor_x->dptr(),
                          tensor_x->shape().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    } else {
      user_op::Tensor* tensor_tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      T* temp_storage = static_cast<T*>(tensor_tmp->mut_dptr());
      NdarrayUtil<device_type, T>::ReduceSum(
          ctx->device_ctx(),
          XpuVarNdarray<T>(CreateReducedShape(tensor_x->shape(), {axis.begin(), axis.end()}),
                           tensor_y->mut_dptr<T>()),
          XpuVarNdarray<const T>(tensor_x->shape(), tensor_x->mut_dptr<T>(),
                                 tensor_x->shape().NumAxes()),
          XpuVarNdarray<T>(tensor_x->shape(), temp_storage, tensor_x->shape().NumAxes()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_SUM_LIKE_KERNEL(device, data_type_pair)                             \
  REGISTER_USER_KERNEL("reduce_sum_like")                                                   \
      .SetCreateFn<ReduceSumLikeOpKernel<device, OF_PP_PAIR_FIRST(data_type_pair)>>()       \
      .SetIsMatchedHob(user_op::HobDeviceType() == device                                   \
                       & user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(data_type_pair)) \
      .SetInferTmpSizeFn(ReduceSumLikeInferTmpSize);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_SUM_LIKE_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
