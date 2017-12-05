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
  void GetSumFromSrcBlobsToDstBlob(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)>,
                                   const std::vector<std::string>& src_bns,
                                   const std::string& dst_bn) const;
  void BoxingCopy(const KernelCtx& ctx, bool is_data_id, Blob* src_blob,
                  Blob* dst_blob, const int64_t src_offset,
                  const int64_t dst_offset, size_t copy_bytesize,
                  bool need_swap) const;
  void CopyDataId(const KernelCtx& ctx, std::vector<Blob*>& src_blobs,
                  std::vector<Blob*>& dst_blobs, const int32_t src_concat_axis,
                  const int32_t dst_split_axis) const;
  void DoUnequalAxisCopy(const KernelCtx& ctx, std::vector<Blob*>& src_blobs,
                         std::vector<Blob*>& dst_blobs,
                         const BoxingInfo& src_info, const BoxingInfo& dst_info,
                         bool need_swap) const;
  void BoxingCopyForUnequalAxis(const KernelCtx& ctx,
                                std::vector<Blob*>& src_blobs,
                                std::vector<Blob*>& dst_blobs,
                                const int32_t concat_axis,
                                const int32_t split_axis) const;
  void BoxingCopyForEqualAxis(const KernelCtx& ctx,
                              std::vector<Blob*>& src_blobs,
                              std::vector<Blob*>& dst_blobs,
                              const int32_t axis) const;
  void CopyFromSrcBlobs2DstBlobs(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)>,
                                 const std::vector<std::string>& src_bns,
                                 const std::vector<std::string>& dst_bns,
                                 const int32_t src_concat_axis,
                                 const int32_t dst_split_axis) const;
  void CopyFromFirstBlob2OtherBlobs(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)>,
                                    const std::vector<std::string>& obns) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
