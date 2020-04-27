#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T>
class CpuIsOrderKernel final : public user_op::OpKernel {
 public:
  CpuIsOrderKernel() = default;
  ~CpuIsOrderKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::string& order_type = ctx->GetAttr<std::string>("order_type");
    const int32_t elem_cnt = in->shape().elem_cnt();
    const bool is_strictly_inc = order_type == "STRICTLY_INCREASING";
    const bool is_non_dec = order_type == "NON_DECREASING";
    const T* x_ptr = in->dptr<T>();
    // use int8 1 as true; int8 0 as false
    int8_t* y_ptr = out->mut_dptr<int8_t>();
    y_ptr[0] = 1;
    auto TryForEach = [&](const int8_t (*Cmp)(const T, const T)) {
      if (elem_cnt >= 2) {
        FOR_RANGE(int32_t, i, 0, elem_cnt - 1) {
          if (Cmp(x_ptr[i], x_ptr[i + 1])) {
            y_ptr[0] = 0;
            break;
          }
        }
      }
    };
    if (is_strictly_inc) {
      TryForEach(&BinaryFuncGE<T>::Invoke);
    } else if (is_non_dec) {
      TryForEach(&BinaryFuncGT<T>::Invoke);
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_CPU_IS_ORDER_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("is_order")                                                    \
      .SetCreateFn<CpuIsOrderKernel<dtype>>()                                         \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && in_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_IS_ORDER_KERNEL(float)
REGISTER_CPU_IS_ORDER_KERNEL(double)
REGISTER_CPU_IS_ORDER_KERNEL(int32_t)
REGISTER_CPU_IS_ORDER_KERNEL(int64_t)

}  // namespace oneflow
