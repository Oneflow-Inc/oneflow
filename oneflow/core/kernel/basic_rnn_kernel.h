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
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void ForwardDataId(const KernelCtx&,
                     std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void VirtualInitModelBlobsWithRandomSeed(
      const KernelCtx&, std::mt19937,
      std::function<Blob*(const std::string&)>) const override;
  void VirtualInitModelBlobsWithDir(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelTmpBlobs(
      const KernelCtx& ctx, const ParallelContext* parallel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class BasicRnnKernelUtil final {
 public:
  static void Add(DeviceCtx* ctx, int64_t n, const T* x, const T* y, T* z);
  static void Tanh(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void ComputePlusOutDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                                 const T* ht_diff, T* plus_out_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_
