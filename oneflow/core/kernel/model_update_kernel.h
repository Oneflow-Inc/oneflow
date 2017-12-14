#ifndef ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class MdUpdateKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdateKernel);
  ~MdUpdateKernel() = default;

  void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    auto tpl = reinterpret_cast<std::tuple<int64_t, const Blob*>*>(ctx.other);
    UpdateModel(ctx.device_ctx, std::get<1>(*tpl), std::get<0>(*tpl),
                BnInOp2Blob);
  }

 protected:
  MdUpdateKernel() = default;
  virtual void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, int64_t next_model_vid,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
