#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

template<DeviceType device, typename T>
class BroadcastDivGradKernel final : public user_op::OpKernel {
 public:
  BroadcastDivGradKernel() = default;
  ~BroadcastDivGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* b_blob = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* db_blob = ctx->Tensor4ArgNameAndIndex("db", 0);

    const int64_t num_axes = dy_blob->shape().NumAxes();
    XpuVarNdarray<const T> dy(dy_blob, num_axes);
    XpuVarNdarray<const T> const_tmp(dy.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dy.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device, T>::BroadcastDiv(ctx->device_ctx(), tmp,
                                         XpuVarNdarray<const T>(y_blob, num_axes),
                                         XpuVarNdarray<const T>(b_blob, num_axes));
    NdarrayUtil<device, T>::BroadcastMul(ctx->device_ctx(), tmp, dy, const_tmp);
    NdarrayUtil<device, T>::ReduceSum(ctx->device_ctx(), XpuVarNdarray<T>(db_blob, num_axes),
                                      const_tmp, tmp);
    NdarrayUtil<device, T>::InplaceNegative(ctx->device_ctx(), XpuVarNdarray<T>(db_blob, num_axes));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BROADCAST_DIV_GRAD_KERNEL(device, dtype_pair)                      \
  REGISTER_USER_KERNEL("broadcast_div_grad")                                        \
      .SetCreateFn<BroadcastDivGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()  \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0); \
        return ctx.device_type() == device                                          \
               && x_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);             \
      })                                                                            \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                  \
        user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);           \
        const DataType& data_type = y->data_type();                                 \
        const int64_t elem_cnt = y->shape().elem_cnt();                             \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));         \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_DIV_GRAD_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
