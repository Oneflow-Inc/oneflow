#ifndef ONEFLOW_CORE_KERNEL_POD_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POD_BOXING_KERNEL_H_

#include "oneflow/core/kernel/boxing_kernel.h"

namespace oneflow {

template<typename T>
class PodBoxingKernel final : public BoxingKernel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PodBoxingKernel);
  PodBoxingKernel() = default;
  ~PodBoxingKernel() = default;

 protected:
  const BoxingOpConf& boxing_conf() const override {
    return this->op_conf().pod_boxing_conf().boxing_conf();
  }
  const PbRpf<std::string>& InputBns() const override { return this->op_attribute().input_bns(); }
  const PbRpf<std::string>& OutputBns() const override { return this->op_attribute().output_bns(); }

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POD_BOXING_KERNEL_H_
