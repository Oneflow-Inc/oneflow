#ifndef ONEFLOW_CORE_KERNEL_RNN_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RNN_DATA_LOADER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename IntegerT>
class RnnDataLoaderKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnDataLoaderKernel);
  RnnDataLoaderKernel() = default;
  ~RnnDataLoaderKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
  void VirtualKernelInit(const ParallelContext*) override;

  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RNN_DATA_LOADER_KERNEL_H_
