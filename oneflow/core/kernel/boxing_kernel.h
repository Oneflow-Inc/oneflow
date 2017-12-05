#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<typename T>
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
  void DoUnequalAxisCopy(const KernelCtx&, std::vector<Blob*>&,
                         std::vector<Blob*>&, const BoxingInfo&,
                         const BoxingInfo&, bool) const;
  void BoxingCopyForUnequalAxis(const KernelCtx&, std::vector<Blob*>&,
                                std::vector<Blob*>&, const int32_t,
                                const int32_t) const;
  void BoxingCopyForEqualAxis(const KernelCtx&, std::vector<Blob*>&,
                              std::vector<Blob*>&, const int32_t) const;
  void CopyFromSrcBlobs2DstBlobs(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)>,
                                 const std::vector<std::string>&,
                                 const std::vector<std::string>&, const int32_t,
                                 const int32_t) const;
  void CopyFromFirstBlob2OtherBlobs(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)>,
                                    const std::vector<std::string>&) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
