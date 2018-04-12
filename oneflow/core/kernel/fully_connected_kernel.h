#ifndef ONEFLOW_CORE_KERNEL_FULLY_CONNECTED_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FULLY_CONNECTED_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class FullyConnectedKernel final
    : public KernelIfWithActivation<device_type, T>,
      public KernelIfWithModel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FullyConnectedKernel);
  FullyConnectedKernel() = default;
  ~FullyConnectedKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void InitPureModelTmpBlobs(
      DeviceCtx*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937*,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FULLY_CONNECTED_KERNEL_H_
