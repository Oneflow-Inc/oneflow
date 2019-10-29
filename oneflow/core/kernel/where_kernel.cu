#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename CondType, typename T>
__global__ void GpuForward(const int64_t elem_cnt, const CondType* condition_ptr, const T* lhs_ptr,
                           const T* rhs_ptr, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_ptr[i] = static_cast<bool>(condition_ptr[i]) ? lhs_ptr[i] : rhs_ptr[i];
  }
}

}  // namespace

template<typename CondType, typename T>
class WhereGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereGpuKernel);
  WhereGpuKernel() = default;
  ~WhereGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* condition_blob = BnInOp2Blob("condition");
    const Blob* lhs_blob = BnInOp2Blob("lhs");
    const Blob* rhs_blob = BnInOp2Blob("rhs");
    Blob* out_blob = BnInOp2Blob("out");
    const auto& shape = condition_blob->shape();
    CHECK_EQ(lhs_blob->shape(), shape);
    CHECK_EQ(rhs_blob->shape(), shape);
    CHECK_EQ(out_blob->shape(), shape);
    const int64_t elem_cnt = shape.elem_cnt();
    GpuForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                 ctx.device_ctx->cuda_stream()>>>(elem_cnt, condition_blob->dptr<CondType>(),
                                                  lhs_blob->dptr<T>(), rhs_blob->dptr<T>(),
                                                  out_blob->mut_dptr<T>());
  }
};

#define MAKE_ENTRY(cond_type_pair, value_type_pair)                                        \
  NEW_REGISTER_KERNEL(                                                                     \
      OperatorConf::kWhereConf,                                                            \
      WhereGpuKernel<OF_PP_PAIR_FIRST(cond_type_pair), OF_PP_PAIR_FIRST(value_type_pair)>) \
      .SetIsMatchedPred([](const KernelConf& conf) {                                       \
        return (DeviceType::kGPU == conf.op_attribute().op_conf().device_type())           \
               && (OF_PP_PAIR_SECOND(cond_type_pair) == conf.where_conf().cond_type())     \
               && (OF_PP_PAIR_SECOND(value_type_pair) == conf.where_conf().value_type());  \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, INT_DATA_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
