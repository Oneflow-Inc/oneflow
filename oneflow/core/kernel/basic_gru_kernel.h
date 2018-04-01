#ifndef ONEFLOW_CORE_KERNEL_BASIC_GRU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_GRU_KERNEL_H_

#include "oneflow/core/kernel/recurrent_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
using FwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, T*);

template<DeviceType device_type, typename T>
using BwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, const T*,
                                  const T*, T*);

template<DeviceType device_type, typename T>
class BasicGruKernel final : public RecurrentKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicGruKernel);
  BasicGruKernel() = default;
  ~BasicGruKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  const PbMessage& GetRecurrentOpConf() const override;
  bool HasInitHiddenInitializer() const override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void VirtualInitModelBlobsWithRandomSeed(
      const KernelCtx&, std::mt19937,
      std::function<Blob*(const std::string&)>) const override;
  void VirtualInitModelBlobsWithDir(
      const KernelCtx&, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelTmpBlobs(
      const KernelCtx& ctx, const ParallelContext* parallel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  FwActivationFunc<device_type, T> activation_fw_func_;
  BwActivationFunc<device_type, T> activation_bw_func_;
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
      Blob* dandidate_out, Blob* reset_out, Blob* temp_data,
      FwActivationFunc<device_type, T> activation_fw_func_);
  static void ComputeOutForward(const KernelCtx& ctx, const Blob* hidden,
                                Blob* candidate_out, Blob* temp_data,
                                Blob* update_out, Blob* out);
  static void ComputeTmpActivationDataDiff(
      const KernelCtx& ctx, Blob* out_diff, Blob* hiddden, Blob* update_d_diff,
      Blob* candidate_d_diff, Blob* tmp_data, Blob* reset_d_diff,
      BwActivationFunc<device_type, T> activation_bw_func_,
      std::function<Blob*(const std::string&)> BnInOp2Blob);
  static void ComputeWeightDiff(const KernelCtx& ctx, const Blob* in_data,
                                Blob* hidden, Blob* out_diff,
                                Blob* i2h_weight_diff, Blob* h2h_weight_diff);
  static void ComputeBiasDiff(
      const KernelCtx& ctx, Blob* reset_d_diff, Blob* update_d_diff,
      Blob* candidate_d_diff, Blob* bias_r_diff, Blob* bias_z_diff,
      Blob* bias_diff, std::function<Blob*(const std::string&)> BnInOp2Blob);
  static void ComputeInDiff(
      const KernelCtx& ctx, Blob* reset_d_diff, Blob* update_d_diff,
      Blob* candidate_d_diff, Blob* in_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob);
  static void ComputeHiddenDiff(
      const KernelCtx& ctx, Blob* hidden, Blob* hidden_diff, Blob* tmp_data,
      Blob* candidate_d_diff, Blob* reset_d_diff, Blob* update_d_diff,
      Blob* out_diff, std::function<Blob*(const std::string&)> BnInOp2Blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEK_BASIC_GRU_KERNEL_H_
