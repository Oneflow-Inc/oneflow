#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class FakeConsumeKernel : public KernelIf<device_type> {
 public:
  FakeConsumeKernel() = default;
  virtual ~FakeConsumeKernel() = default;

 private:
  typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
  void Forward(const KernelCtx& ctx,
               BnInOp2BlobFunc BnInOp2Blob) const override {
    // Do nothing for fake consumer
  };
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFakeConsumeConf, FakeConsumeKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
