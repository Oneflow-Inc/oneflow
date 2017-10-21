#ifndef ONEFLOW_CORE_KERNEL_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DATA_LOADER_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/persistence/ofb_in_stream.h"

namespace oneflow {

template<typename T>
class DataLoaderKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderKernel);
  DataLoaderKernel() = default;
  ~DataLoaderKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
  void InitOfbInStream(const KernelCtx&) const;
  mutable std::unique_ptr<OfbInStream> ofb_in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_LOADER_KERNEL_H_
