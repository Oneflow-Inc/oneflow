#include "oneflow/core/kernel/upsample_nearest_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void UpsampleNearest<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  auto kernel_conf = this->kernel_conf();
  UpsampleNearestUtil<device_type, T>::Forward(ctx, this->op_conf().upsample_nearest_conf(),
                                               in_blob, out_blob);
}

template<DeviceType device_type, typename T>
void UpsampleNearest<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  UpsampleNearestUtil<device_type, T>::Backward(ctx, this->op_conf().upsample_nearest_conf(),
                                                out_diff_blob, in_diff_blob);
}

template<typename T>
struct UpsampleNearestUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const UpsampleNearestOpConf& conf, const Blob* in_blob,
                      Blob* out_blob) {
    UNIMPLEMENTED();
  }
  static void Backward(const KernelCtx& ctx, const UpsampleNearestOpConf& conf,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kUpsampleNearestConf, UpsampleNearest,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
