#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class CpuArgSortKernel final : public user_op::OpKernel {
 public:
  CpuArgSortKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  CpuArgSortKernel() = default;
  ~CpuArgSortKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const std::string& direction = ctx->GetAttr<std::string>("direction");
    const bool is_ascending = direction == "ASCENDING";
    const bool is_descending = direction == "DESCENDING";
    FOR_RANGE(int32_t, i, 0, instance_num) {
      const T* in_ptr_i = in->dptr<T>() + i * instance_size;
      int32_t* out_ptr_i = out->mut_dptr<int32_t>() + i * instance_size;
      std::iota(out_ptr_i, out_ptr_i + instance_size, 0);
      auto comp = [&](const int32_t lhs, const int32_t rhs) {
        const T l = in_ptr_i[lhs];
        const T r = in_ptr_i[rhs];
        if (l == r) {
          return lhs < rhs;
        } else {
          if (is_ascending) {
            return l < r;
          } else if (is_descending) {
            return l > r;
          } else {
            UNIMPLEMENTED();
          }
        }
      };
      std::sort(out_ptr_i, out_ptr_i + instance_size, comp);
    }
  };
};

#define REGISTER_CPU_ARG_SORT_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("arg_sort")                                                          \
      .SetCreateFn(                                                                         \
          [](user_op::KernelInitContext* ctx) { return new CpuArgSortKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                          \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0);       \
        return ctx.device_type() == DeviceType::kCPU                                        \
               && in_desc->data_type() == GetDataType<dtype>::value;                        \
      });

REGISTER_CPU_ARG_SORT_KERNEL(float)
REGISTER_CPU_ARG_SORT_KERNEL(double)
REGISTER_CPU_ARG_SORT_KERNEL(int32_t)
REGISTER_CPU_ARG_SORT_KERNEL(int64_t)

}  // namespace oneflow
