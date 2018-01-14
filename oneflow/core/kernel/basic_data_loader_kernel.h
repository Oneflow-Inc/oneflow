#ifndef ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_

#include "oneflow/core/actor/source_compute_actor.h"
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
  void ReadOnePieceToBlob(SourceCompActor::DataLoadStatus*, Blob*) const;
  void ReadOneColFromBufferToOutBlob(const KernelCtx&, const Blob*,
                                     Blob*) const;
  const char* ReadOneDataId(const char*, Blob*, int64_t) const;
  int32_t ReadOneDataContent(const char*, Blob*, int64_t) const;

  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_
