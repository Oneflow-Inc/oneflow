#ifndef ONEFLOW_CORE_KERNEL_EMBEDDING_LOOKUP_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_EMBEDDING_LOOKUP_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class EmbeddingLookupKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupKernel);
  EmbeddingLookupKernel() = default;
  ~EmbeddingLookupKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class EmbeddingLookupKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupKernelUtil);
  EmbeddingLookupKernelUtil() = delete;

  static void Forward(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_EMBEDDING_LOOKUP_KERNEL_H_
