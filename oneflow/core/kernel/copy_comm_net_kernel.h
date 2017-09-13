#ifndef ONEFLOW_CORE_KERNEL_COPY_COMM_NET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_COMM_NET_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

class CopyCommNetKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetKernel);
  CopyCommNetKernel() = default;
  ~CopyCommNetKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override {}

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_COMM_NET_KERNEL_H_
