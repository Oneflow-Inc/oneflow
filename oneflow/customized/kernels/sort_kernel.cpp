#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class CpuSortKernel final : public user_op::OpKernel {
 public:
  CpuSortKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  CpuSortKernel() = default;
  ~CpuSortKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    Memcpy<DeviceType::kCPU>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                             in->shape().elem_cnt() * sizeof(T));
    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const std::string& direction = ctx->GetAttr<std::string>("direction");
    const bool is_ascending = direction == "ASCENDING";
    const bool is_descending = direction == "DESCENDING";
    FOR_RANGE(int32_t, i, 0, instance_num) {
      T* out_ptr_i = out->mut_dptr<T>() + i * instance_size;
      if (is_ascending) {
        std::sort(out_ptr_i, out_ptr_i + instance_size, std::less<T>());
      } else if (is_descending) {
        std::sort(out_ptr_i, out_ptr_i + instance_size, std::greater<T>());
      } else {
        UNIMPLEMENTED();
      }
    }
  };
};

#define REGISTER_CPU_SORT_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("sort")                                                          \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                 \
        return new CpuSortKernel<dtype>(ctx);                                           \
      })                                                                                \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {             \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0); \
        return ctx.device_type() == DeviceType::kCPU                                    \
               && out_desc->data_type() == GetDataType<dtype>::value;                   \
      });

REGISTER_CPU_SORT_KERNEL(float)
REGISTER_CPU_SORT_KERNEL(double)
REGISTER_CPU_SORT_KERNEL(int32_t)
REGISTER_CPU_SORT_KERNEL(int64_t)

}  // namespace oneflow
