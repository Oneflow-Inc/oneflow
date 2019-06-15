#ifndef ONEFLOW_CORE_KERNEL_LOCAL_SCATTER_ND_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOCAL_SCATTER_ND_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class LocalScatterNdUpdateKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalScatterNdUpdateKernel);
  LocalScatterNdUpdateKernel() = default;
  ~LocalScatterNdUpdateKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
struct LocalScatterNdUpdateKernelUtil final {
  static void Forward(DeviceCtx* ctx, int64_t* shape_ptr, const Blob* indices_blob,
                      const Blob* updates_blob, const int64_t num_updates, const int64_t block_size,
                      Blob* out_blob);
  static void Backward(DeviceCtx* ctx, const Blob* out_diff_blob, int64_t* shape_ptr,
                       const Blob* indices_blob, const int64_t num_updates,
                       const int64_t block_size, Blob* updates_diff_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_SCATTER_ND_UPDATE_KERNEL_H_
