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

template<DeviceType device_type, typename T>
struct BasicGruKernelUtil {
  static void ComputeGateForward(const KernelCtx& ctx, const Blob* in_data,
                                 const Blob* hidden,
                                 const Blob* bias_multiplier,
                                 const Blob* i2h_weight, const Blob* h2h_weight,
                                 const Blob* bias, Blob* gate_data,
                                 Blob* gate_out);
  static void ComputeCandidateHiddenForward(
      const KernelCtx& ctx, const Blob* in_data, const Blob* hidden,
      const Blob* bias_multiplier, const Blob* i2h_weight,
      const Blob* h2h_weight, const Blob* bias, Blob* candidate_data,
      Blob* dandidate_out, Blob* reset_out, Blob* temp_data);
  static void ComputePlusOutForward(const KernelCtx& ctx, const Blob* hidden,
                                    Blob* candidate_out, Blob* temp_data,
                                    Blob* update_out, Blob* plus_out);
  static void ComputeWeightDiff(const KernelCtx& ctx, const Blob* in_data,
                                Blob* hidden, Blob* out_diff,
                                Blob* i2h_weight_diff, Blob* h2h_weight_diff);
  static void ComputeTmpModelDiff(
      const KernelCtx& ctx, const Blob* candidate_out,
      const Blob* candidate_data, const Blob* update_out,
      const Blob* update_data, const Blob* reset_out, const Blob* reset_data,
      Blob* hiddden, Blob* candidate_o_diff, Blob* candidate_d_diff,
      Blob* update_o_diff, Blob* update_o_bran_diff, Blob* update_d_diff,
      Blob* reset_o_diff, Blob* reset_d_diff, Blob* plus_out_diff);
  static void ComputeHiddenDiff(const KernelCtx& ctx, const Blob* h2h_weight_r,
                                const Blob* h2h_weight_z,
                                const Blob* h2h_weight, const Blob* reset_out,
                                const Blob* update_out, Blob* hidden,
                                Blob* hidden_diff, Blob* candidate_d_diff,
                                Blob* reset_d_diff, Blob* update_d_diff,
                                Blob* plus_out_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEK_BASIC_GRU_KERNEL_H_
