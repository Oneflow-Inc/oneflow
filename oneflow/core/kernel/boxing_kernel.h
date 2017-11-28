#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxingKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
  void GetSumFromSrcBlobsToDstBlob(const KernelCtx&,
                                   std::function<Blob*(const std::string&)>,
                                   const std::vector<std::string>&,
                                   const std::string&) const;
  void BoxingCopy(const KernelCtx&, bool, Blob*, Blob*, const int64_t,
                  const int64_t, size_t, bool) const;
  void CopyDataId(const KernelCtx&, std::vector<Blob*>&, std::vector<Blob*>&,
                  const int32_t, const int32_t) const;
  void InferCopyRulesFromUnequalAxis(const KernelCtx&, std::vector<Blob*>&,
                                     std::vector<Blob*>&, const int32_t,
                                     const int32_t) const;
  void InferCopyRulesFromEqualAxis(const KernelCtx&, std::vector<Blob*>&,
                                   std::vector<Blob*>&, const int32_t) const;
  void CopyFromSrc2Dst(const KernelCtx& ctx,
                       std::function<Blob*(const std::string&)>,
                       const std::vector<std::string>&,
                       const std::vector<std::string>&, const int32_t,
                       const int32_t) const;
  void FwCloneData(const KernelCtx& ctx,
                   std::function<Blob*(const std::string&)>,
                   const std::vector<std::string>&) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
