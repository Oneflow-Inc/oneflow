#ifndef ONEFLOW_CORE_KERNEL_PRELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRELU_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PReluKernel final : public KernelIfWithModel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluKernel);
  PReluKernel() = default;
  ~PReluKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx*, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct PReluKernelUtil {
  static void Forward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                      const Blob* alpha_blob, Blob* out_blob);
  static void Backward(const KernelCtx& ctx, const PReluOpConf& conf,
                       const PbRf<int32_t>& permutation, const Blob* in_blob,
                       const Blob* alpha_blob, const Blob* out_diff_blob, Blob* bw_buf_blob,
                       Blob* in_diff_blob, Blob* alpha_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRELU_KERNEL_H_
