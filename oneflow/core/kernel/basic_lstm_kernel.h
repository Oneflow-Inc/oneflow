#ifndef ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

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
      DeviceCtx*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void VirtualInitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void VirtualInitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937*,
      std::function<Blob*(const std::string&)>) const;
  void VirtualKernelInit(const ParallelContext*) override;

 private:
  bool need_external_h0_;
  bool need_external_c0_;
};

template<DeviceType device_type, typename T>
struct BasicLstmKernelUtil {
  static void ComputeForwardGateOut(const KernelCtx& ctx, Blob* gate_out,
                                    const Blob* i2h_weight, const Blob* hidden,
                                    const Blob* h2h_weight, const Blob* input);

  static void ComputeBackwardCellOutDiff(const KernelCtx& ctx, Blob* cell_out,
                                         Blob* cell_out_diff, Blob* o_out,
                                         Blob* rec_out_diff, Blob* out_diff);
  static void ComputeBackwardWeightDiff(const KernelCtx& ctx, const Blob* input,
                                        Blob* gate_out_diff, const Blob* hidden,
                                        Blob* h2h_weight_diff,
                                        Blob* i2h_weight_diff);
};

}  //  namespace oneflow

#endif  //  ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
