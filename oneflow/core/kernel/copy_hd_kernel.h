#ifndef ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

class CopyHdKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  cudaMemcpyKind fw_kind_;
  cudaMemcpyKind bw_kind_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
