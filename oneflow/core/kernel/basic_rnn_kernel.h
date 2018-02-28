#ifndef ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_

#include "oneflow/core/kernel/recurrent_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BasicRnnKernel final : public RecurrentKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicRnnKernel);
  BasicRnnKernel() = default;
  ~BasicRnnKernel() = default;

 private:
  const PbMessage& GetRecurrentOpConf() const override;
  bool HasInitHiddenInitializer() const override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelTmpBlobs(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void VirtualInitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937*,
      std::function<Blob*(const std::string&)>) const override;
  void VirtualInitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class BasicRnnKernelUtil final {
 public:
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* out,
                              const T* out_diff, const T* rec_out_diff,
                              T* plus_out_diff);
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* out,
                                 const T* out_diff, const T* rec_out_diff,
                                 T* plus_out_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_
