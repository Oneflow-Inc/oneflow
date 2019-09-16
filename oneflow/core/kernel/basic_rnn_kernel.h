#ifndef ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_

#include "oneflow/core/kernel/recurrent_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
using FwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, T*);

template<DeviceType device_type, typename T>
using BwActivationFunc = void (*)(DeviceCtx* ctx, int64_t n, const T*, const T*, const T*, T*);

template<DeviceType device_type, typename T>
class BasicRnnKernel final : public RecurrentKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicRnnKernel);
  BasicRnnKernel() = default;
  ~BasicRnnKernel() = default;

 private:
  void VirtualKernelInit() override;
  const PbMessage& GetRecurrentOpConf() const override;
  bool HasInitHiddenInitializer() const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  FwActivationFunc<device_type, T> activation_fw_func_;
  BwActivationFunc<device_type, T> activation_bw_func_;
  BwActivationFunc<device_type, T> last_colnum_activation_bw_func_;
};

template<DeviceType device_type, typename T>
struct BasicRnnKernelUtil {
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                              const T* rec_out_diff, T* plus_out_diff);
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                                 const T* rec_out_diff, T* plus_out_diff);
  static void ComputeReluDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                              const T* rec_out_diff, T* plus_out_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BASIC_RNN_KERNEL_H_
