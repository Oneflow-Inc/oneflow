#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ZeroLikeKernel final : public user_op::OpKernel {
 public:
  ZeroLikeKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ZeroLikeKernel() = default;
  ~ZeroLikeKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memset<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), 0,
                        out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()));
  };
};

#define REGISTER_ZERO_LIKE_KERNEL(device_type_v, data_type_pair)                         \
  REGISTER_USER_KERNEL("zero_like")                                                      \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                  \
        return new ZeroLikeKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair)>(ctx); \
      })                                                                                 \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {              \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);  \
        if (ctx.device_type() == device_type_v                                           \
            && out_desc->data_type() == OF_PP_PAIR_SECOND(data_type_pair)) {             \
          return true;                                                                   \
        }                                                                                \
        return false;                                                                    \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ZERO_LIKE_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
