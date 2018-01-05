#ifndef ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class BasicDataLoaderKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicDataLoaderKernel);
  BasicDataLoaderKernel() = default;
  ~BasicDataLoaderKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
  void VirtualKernelInit(const ParallelContext*) override;

  void ReadOnePieceToBuffer(Blob* buffer_blob) const;

  void CopyDataToOutBlob(Blob* buffer_blob, Blob* out_blob) const;

  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_
