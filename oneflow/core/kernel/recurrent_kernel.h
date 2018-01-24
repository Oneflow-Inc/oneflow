#ifndef ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RecurrentKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentKernel);
  ~RecurrentKernel() = default;

  bool Ish0Model() const;
  Blob* GetHiddenBlob(std::function<Blob*(const std::string&)>) const;

  void InitModelBlobsWithRandomSeed(
      const KernelCtx& ctx, std::mt19937 random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)>) const override;

 protected:
  RecurrentKernel() = default;

  virtual void VirtualInitModelBlobsWithRandomSeed(
      const KernelCtx& ctx, std::mt19937 random_seed_gen,
      std::function<Blob*(const std::string&)>) const {}
  virtual void VirtualInitModelBlobsWithDir(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)>) const {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_
