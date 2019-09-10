#ifndef ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BiasAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BiasAddKernel);
  BiasAddKernel() = default;
  ~BiasAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
struct BiasAddUtil {
  static void BiasAddNCX(DeviceCtx* ctx, const Shape& shape, const int32_t bias_axis,
                         const T* input, const T* bias, T* output);
};

template<>
struct BiasAddUtil<DeviceType::kGPU, float16> {
  static void BiasAddNCX(DeviceCtx* ctx, const Shape& shape, const int32_t bias_axis,
                         const float16* input, const float16* bias, float16* output);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_
