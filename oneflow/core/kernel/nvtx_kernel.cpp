#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"
#include "oneflow/core/job/nvtx_ctx.h"

namespace oneflow {

template<DeviceType device_type>
class NvtxRangeStartKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangeStartKernel);
  NvtxRangeStartKernel() = default;
  ~NvtxRangeStartKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    const nvtxRangeId_t id = nvtxRangeStartA(this->op_conf().nvtx_range_start_conf().msg().c_str());
    Global<NvtxCtx>::Get()->PutRangeId(this->op_conf().nvtx_range_start_conf().msg(), id);
  }
};

template<DeviceType device_type>
class NvtxRangeEndKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangeEndKernel);
  NvtxRangeEndKernel() = default;
  ~NvtxRangeEndKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    nvtxRangeEnd(Global<NvtxCtx>::Get()->PopRangeId(this->op_conf().nvtx_range_end_conf().msg()));
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNvtxRangeStartConf, NvtxRangeStartKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNvtxRangeEndConf, NvtxRangeEndKernel);

}  // namespace oneflow
