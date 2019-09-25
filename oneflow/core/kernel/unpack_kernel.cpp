#include "oneflow/core/kernel/unpack_kernel.h"
#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;
  PackKernelUtil<device_type>::Unpack(ctx.device_ctx, out_index, total_unpack_num,
                                      BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type>
void UnpackKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kUnpackConf, UnpackKernel);

}  // namespace oneflow
