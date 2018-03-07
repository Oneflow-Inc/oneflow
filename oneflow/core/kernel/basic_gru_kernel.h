#ifndef ONEFLOW_CORE_KERNEL_BASIC_GRU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_GRU_KERNEL_H_

#include "oneflow/core/kernel/recurrent_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BasicGruKernel final : public RecurrentKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicGruKernel);
  BasicGruKernel() = default;
  ~BasicGruKernel() = default;

 private:
  const PbMessage& GetRecurrentOpConf() const override;
  bool HasInitHiddenInitializer() const override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void InitPureModelTmpBlobs(
      DeviceCtx*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void VirtualInitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937*,
      std::function<Blob*(const std::string&)>) const override;
  void VirtualInitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif //ONEFLOW_CORE_KERNEK_BASIC_GRU_KERNEL_H_

