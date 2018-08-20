#ifndef ONEFLOW_CORE_KERNEL_PB_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PB_BOXING_KERNEL_H_

#include "oneflow/core/kernel/boxing_kernel.h"

namespace oneflow {

template<typename T>
class PbBoxingKernel final : public BoxingKernel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PbBoxingKernel);
  PbBoxingKernel() = default;
  ~PbBoxingKernel() = default;

 protected:
  const BoxingOpConf& boxing_conf() const override {
    return this->op_conf().pb_boxing_conf().boxing_conf();
  }
  const PbRpf<std::string>& InputBns() const override {
    return this->op_attribute().pb_input_bns();
  }
  const PbRpf<std::string>& OutputBns() const override {
    return this->op_attribute().pb_output_bns();
  }

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PB_BOXING_KERNEL_H_
