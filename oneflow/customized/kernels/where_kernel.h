#ifndef ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
struct WhereFunctor {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const;
};

template<typename T, typename CondT>
OF_DEVICE_FUNC void DoWhere(const int64_t elem_cnt, const CondT* condition, const T* lhs,
                            const T* rhs, T* out) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = static_cast<bool>(condition[i]) ? lhs[i] : rhs[i]; }
}

template<DeviceType device_type, typename T, typename CondT>
class WhereKernel final : public user_op::OpKernel {
 public:
  WhereKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

size_t InferWhereTmpBufferSize(oneflow::user_op::InferContext* ctx);

#define REGISTER_WHERE_KERNEL(device_type_v, dtype, ctype)                                     \
  REGISTER_USER_KERNEL("where")                                                                \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                        \
        return new WhereKernel<device_type_v, dtype, ctype>(ctx);                              \
      })                                                                                       \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* cond_desc = ctx.TensorDesc4ArgNameAndIndex("condition", 0); \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);        \
        return ctx.device_type() == device_type_v                                              \
               && cond_desc->data_type() == GetDataType<ctype>::value                          \
               && out_desc->data_type() == GetDataType<dtype>::value;                          \
      })                                                                                       \
      .SetInferTmpSizeFn(InferWhereTmpBufferSize);

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_H_
