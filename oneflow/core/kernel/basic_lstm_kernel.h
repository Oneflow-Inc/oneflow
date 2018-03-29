#ifndef ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
using FwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, T*);

template<DeviceType device_type, typename T>
using BwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, const T*,
                                  const T*, T*);

template<DeviceType device_type, typename T>
class BasicLstmKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicLstmKernel);
  BasicLstmKernel() = default;
  ~BasicLstmKernel() = default;

 private:
  const PbMessage& GetBasicLstmOpConf() const;
  bool HasInitHiddenInitializer() const;
  bool HasInitCellInitializer() const;
  bool NeedExternalH0() const;
  bool NeedExternalC0() const;
  Blob* GetHiddenBlob(std::function<Blob*(const std::string&)>) const;
  Blob* GetHiddenDiffBlob(std::function<Blob*(const std::string&)>) const;
  Blob* GetCellBlob(std::function<Blob*(const std::string&)>) const;
  Blob* GetCellDiffBlob(std::function<Blob*(const std::string&)>) const;

  void ForwardColNum(const KernelCtx&,
                     std::function<Blob*(const std::string&)>) const override;
  void ForwardDataId(const KernelCtx&,
                     std::function<Blob*(const std::string&)>) const override;
  void BackwardColNum(const KernelCtx&,
                      std::function<Blob*(const std::string&)>) const override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void InitPureModelTmpBlobs(
      const KernelCtx&, const ParallelContext* parallel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void VirtualInitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void VirtualInitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937*,
      std::function<Blob*(const std::string&)>) const;
  void VirtualKernelInit(const ParallelContext*) override;

 private:
  FwActivationFunc<device_type, T> activation_fw_func_;
  BwActivationFunc<device_type, T> activation_bw_func_;
  bool need_external_h0_;
  bool need_external_c0_;
};

template<DeviceType device_type, typename T>
struct BasicLstmKernelUtil {
  static void ComputeForwardGateOut(const KernelCtx& ctx,
                                    const Blob* i2h_weight, const Blob* hidden,
                                    const Blob* h2h_weight, const Blob* input,
                                    const Blob* bias_mul, const Blob* bias,
                                    Blob* state_data);

  static void ComputeBackwardCellOutDiff(
      const KernelCtx& ctx, const Blob* rec_out_diff, Blob* candidate_out,
      Blob* cell_out, Blob* cell_out_diff, Blob* o_out, Blob* out_diff,
      BwActivationFunc<device_type, T> acticaiton_bw_func_);
  static void ComputeBackwardWeightDiff(const KernelCtx& ctx, const Blob* input,
                                        const Blob* hidden, Blob* gate_out_diff,
                                        Blob* h2h_weight_diff,
                                        Blob* i2h_weight_diff);
  static void ComputeBackwardHiddenDiff(
      const KernelCtx& ctx, const Blob* h2h_f_weight, const Blob* h2h_i_weight,
      const Blob* h2h_c_weight, const Blob* h2h_o_weight, Blob* f_data_diff,
      Blob* i_data_blob, Blob* c_data_diff, Blob* o_data_diff,
      Blob* hidden_diff);
};

}  //  namespace oneflow

#endif  //  ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
