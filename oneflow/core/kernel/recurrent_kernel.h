#ifndef ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RecurrentKernel : public KernelIfWithModel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentKernel);
  virtual ~RecurrentKernel() = default;

 protected:
  RecurrentKernel() = default;

  virtual const PbMessage& GetRecurrentOpConf() const = 0;
  virtual bool HasInitHiddenInitializer() const = 0;
  bool NeedExternalH0() const;
  Blob* GetHiddenBlob(std::function<Blob*(const std::string&)>) const;
  Blob* GetHiddenDiffBlob(std::function<Blob*(const std::string&)>) const;

  void ForwardColNum(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void ForwardDataId(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void BackwardColNum(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx*, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)>) const override;
  void VirtualKernelInit(const ParallelContext*) override;
  virtual void VirtualInitModelBlobsWithRandomSeed(DeviceCtx*, std::mt19937* random_seed_gen,
                                                   std::function<Blob*(const std::string&)>) const {
  }
  virtual void VirtualInitModelBlobsWithDir(DeviceCtx*, int32_t part_id, int32_t part_num,
                                            const std::string& model_load_dir,
                                            std::function<Blob*(const std::string&)>) const {}

 private:
  bool need_external_h0_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_
