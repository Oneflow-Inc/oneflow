#ifndef ONEFLOW_CORE_KERNEL_RNN_LOOKUP_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RNN_LOOKUP_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<typename IntegerType, typename FloatType>
class RnnLookupKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnLookupKernel);
  RnnLookupKernel() = default;
  ~RnnLookupKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RNN_LOOKUP_KERNEL_H_
