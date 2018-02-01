#ifndef ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RecurrentKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentKernel);
  ~RecurrentKernel() = default;

  bool NeedExternalH0() const;
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

#define DECLARE_RECURRENT_KERNEL_CREATOR(x) \
  Kernel* Create##x##Kernel(const KernelConf&);

#define DEFINE_RECCURENT_KERNEL_CREATOR(x)                                   \
  Kernel* Create##x##Kernel(const KernelConf& kernel_conf) {                 \
    static const HashMap<std::string, std::function<Kernel*()>> creators = { \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY,          \
                                         (x##Kernel), DEVICE_TYPE_SEQ,       \
                                         FLOATING_DATA_TYPE_SEQ)};           \
    return creators.at(                                                      \
        GetHashKey(kernel_conf.device_type(), kernel_conf.data_type()))();   \
  }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECURRENT_KERNEL_H_
