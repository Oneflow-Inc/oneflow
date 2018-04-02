#ifndef ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_

#include "oneflow/core/kernel/recurrent_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
using FwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, T*);

template<DeviceType device_type, typename T>
using BwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, const T*,
                                  const T*, T*);

template<DeviceType device_type, typename T>
class BasicLstmKernel : public RecurrentKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicLstmKernel);
  BasicLstmKernel() = default;
  ~BasicLstmKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  const PbMessage& GetRecurrentOpConf() const override;
  bool HasInitHiddenInitializer() const override;
  bool HasInitCellInitializer() const;
  bool NeedExternalC0() const;
  Blob* GetCellBlob(std::function<Blob*(const std::string&)>) const;
  Blob* GetCellDiffBlob(std::function<Blob*(const std::string&)>) const;
  void ForwardColNum(const KernelCtx&,
                     std::function<Blob*(const std::string&)>) const override;
  void BackwardColNum(const KernelCtx&,
                      std::function<Blob*(const std::string&)>) const override;
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
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelTmpBlobs(
      const KernelCtx& ctx, const ParallelContext* parallel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  FwActivationFunc<device_type, T> activation_fw_func_;
  BwActivationFunc<device_type, T> activation_bw_func_;
  bool need_external_c0_;
};

template<DeviceType device_type, typename T>
struct BasicLstmKernelUtil {
  static void ComputeForwardGateOut(const KernelCtx& ctx,
                                    const Blob* i2h_weight, const Blob* hidden,
                                    const Blob* h2h_weight, const Blob* input,
                                    const Blob* bias_mul, const Blob* bias,
                                    Blob* gate_tmp_data);

  static void ComputeRecCellOutDiff(
      const KernelCtx& ctx, Blob* candidate_out, Blob* rec_cell_out_diff,
      BwActivationFunc<device_type, T> acticaiton_bw_func_,
      std::function<Blob*(const std::string&)> BnInOp2Blob);

  static void ComputeActivationDataDiff(
      const KernelCtx& ctx, const Blob* rec_cell_out_diff, Blob* f_data_diff,
      Blob* i_data_diff, Blob* c_data_diff, Blob* o_data_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob,
      BwActivationFunc<device_type, T> acticaiton_bw_func_);

  static void ComputeAllWeightDiff(const KernelCtx& ctx, const Blob* input,
                                   const Blob* hidden, Blob* f_data_diff,
                                   Blob* i_data_diff, Blob* c_data_diff,
                                   Blob* o_data_diff, Blob* f_h2h_diff,
                                   Blob* f_i2h_diff, Blob* i_h2h_diff,
                                   Blob* i_i2h_diff, Blob* c_h2h_diff,
                                   Blob* c_i2h_diff, Blob* o_h2h_diff,
                                   Blob* o_i2h_diff);

  static void ComputeAllBiasDiff(
      const KernelCtx& ctx, Blob* f_data_diff, Blob* i_data_diff,
      Blob* c_data_diff, Blob* o_data_diff, Blob* bias_f_diff,
      Blob* bias_i_diff, Blob* bias_c_diff, Blob* bias_o_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob);

  static void ComputeInDiff(
      const KernelCtx& ctx, Blob* in_diff, Blob* f_data_diff, Blob* i_data_diff,
      Blob* c_data_diff, Blob* o_data_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob);

  static void ComputeHiddenDiff(
      const KernelCtx& ctx, Blob* f_data_diff, Blob* i_data_blob,
      Blob* c_data_diff, Blob* o_data_diff, Blob* hidden_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob);
};

}  //  namespace oneflow

#endif  //  ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
