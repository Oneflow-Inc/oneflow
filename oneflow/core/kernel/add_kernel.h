#ifndef ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AddUtil;

template<DeviceType device_type, typename T>
class AddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddKernel);
  AddKernel() = default;
  ~AddKernel() = default;

 private:
  friend class AddUtil<device_type, T>;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  const PbMessage& GetCustomizedOpConf() const override;

  decltype(make_tuple_from_sequence<7>()) tp_;
};

void HalfGpuAdd(DeviceCtx* ctx, const int64_t n, float16* out_dptr,
                const std::vector<const float16*>& in_dptrs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_
