#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
void cpu_add(const int64_t n, T* out, const std::vector<const T*>& in) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] = in.at(0)[i];
    for (int32_t j = 1; j < in.size(); ++j) { out[i] += in.at(j)[i]; }
  }
}

}  // namespace

template<typename T>
class CpuAddNKernel : public user_op::OpKernel {
 public:
  CpuAddNKernel() = default;
  ~CpuAddNKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->inputs().size();

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();

    std::vector<const T*> in_dptrs(in_num);
    for (int32_t i = 0; i < in_num; ++i) {
      in_dptrs.at(i) = ctx->Tensor4ArgNameAndIndex("in", i)->dptr<T>();
    }

    cpu_add<T>(n, out_dptr, in_dptrs);
  }
};

#define REGISTER_CPU_ADDN_KERNEL(cpp_type, dtype)                                               \
  REGISTER_USER_KERNEL("add_n")                                                                 \
      .SetCreateFn<CpuAddNKernel<cpp_type>>()                                                   \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU                             \
                       & user_op::HobDataType("in", 0) == dtype)                                \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_CPU_ADDN_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
