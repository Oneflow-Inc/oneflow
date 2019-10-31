#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename CondType, typename T>
__global__ void GpuForward(const int64_t elem_cnt, const CondType* cond_dptr, const T* x_dptr,
                           const T* y_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_dptr[i] = (cond_dptr[i] != 0) * x_dptr[i] + (cond_dptr[i] == 0) * y_dptr[i];
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
    const Blob* x_blob = BnInOp2Blob("x");
    const Blob* y_blob = BnInOp2Blob("y");
    Blob* out_blob = BnInOp2Blob("out");
    const int64_t elem_cnt = condition_blob->shape().elem_cnt();
    GpuForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                 ctx.device_ctx->cuda_stream()>>>(elem_cnt, condition_blob->dptr<CondType>(),
                                                  x_blob->dptr<T>(), y_blob->dptr<T>(),
                                                  out_blob->mut_dptr<T>());
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().where_conf(); }
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

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
