#include "oneflow/core/kernel/const_range_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConstRangeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (*output_inited_) { return; }
  const ConstRangeOpConf& range_conf = this->op_conf().const_range_conf();
  const int64_t size = range_conf.size();
  const T start = static_cast<T>(range_conf.start());
  const T stride = static_cast<T>(range_conf.stride());
  ConstRangeKernelUtil<device_type, T>::Fill(ctx.device_ctx, start, size, stride,
                                             BnInOp2Blob("out")->mut_dptr<T>());
  *output_inited_ = true;
}

template<DeviceType device_type, typename T>
const PbMessage& ConstRangeKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().const_range_conf();
}

template<typename T>
struct ConstRangeKernelUtil<DeviceType::kCPU, T> final {
  static void Fill(DeviceCtx* ctx, T start, int64_t size, T stride, T* out);
};

template<typename T>
void ConstRangeKernelUtil<DeviceType::kCPU, T>::Fill(DeviceCtx* ctx, T start, int64_t size,
                                                     T stride, T* out) {
  FOR_RANGE(int64_t, i, 0, size) { out[i] = start + i * stride; }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConstRangeConf, ConstRangeKernel, INT_DATA_TYPE_SEQ);

}  // namespace oneflow
