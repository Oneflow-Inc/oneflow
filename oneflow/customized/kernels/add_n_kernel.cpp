#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
void cpu_add(const int64_t n, T* out, const T* in_0) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i]; }
}
template<typename T>
void cpu_add(const int64_t n, T* out, const T* in_0, const T* in_1) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i]; }
}
template<typename T>
void cpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}
template<typename T>
void cpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2, const T* in_3) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
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
    CHECK_GE(in_num, 2);
    CHECK_LE(in_num, 4)
        << "CpuAddNKernel of add_n op doesn't support number of operands which is bigger than 4.";

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();

    std::vector<const T*> in_dptrs(in_num);
    for (int32_t i = 0; i < in_num; ++i) {
      in_dptrs.at(i) = ctx->Tensor4ArgNameAndIndex("in", i)->dptr<T>();
    }

    switch (in_num) {
      case 2: cpu_add(n, out_dptr, in_dptrs.at(0), in_dptrs.at(1)); break;
      case 3: cpu_add(n, out_dptr, in_dptrs.at(0), in_dptrs.at(1), in_dptrs.at(2)); break;
      case 4:
        cpu_add(n, out_dptr, in_dptrs.at(0), in_dptrs.at(1), in_dptrs.at(2), in_dptrs.at(3));
        break;
      default: LOG(FATAL) << "Not supported input size for CpuAddNKernel: " << in_num; break;
    }
  }
};

#define REGISTER_CPU_ADDN_KERNEL(cpp_type, dtype)                                        \
  REGISTER_USER_KERNEL("add_n").SetCreateFn<CpuAddNKernel<cpp_type>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                         \
        return ctx.device_type() == DeviceType::kCPU                                     \
               && ctx.TensorDesc4ArgNameAndIndex("in", 0)->data_type() == dtype;         \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_CPU_ADDN_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
