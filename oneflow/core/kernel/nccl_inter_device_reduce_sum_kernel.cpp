#include "oneflow/core/kernel/nccl_inter_device_reduce_sum_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void NcclInterDeviceReduceSumKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* fw_buf = BnInOp2Blob("fw_buf");
  Blob* out = BnInOp2Blob("out");
  Memcpy<device_type>(ctx.device_ctx, fw_buf->mut_dptr(), in->dptr(),
                      in->ByteSizeOfDataContentField());
  NcclInterDeviceReduceSumKernelUtil<device_type>::ReduceSum(ctx.device_ctx, fw_buf, fw_buf);
  out->CopyDataContentFrom(ctx.device_ctx, fw_buf);
}

template<DeviceType device_type>
void NcclInterDeviceReduceSumKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* bw_buf = BnInOp2Blob("bw_buf");
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  Memcpy<device_type>(ctx.device_ctx, bw_buf->mut_dptr(), out_diff->dptr(),
                      out_diff->ByteSizeOfDataContentField());
  NcclInterDeviceReduceSumKernelUtil<device_type>::ReduceSum(ctx.device_ctx, bw_buf, bw_buf);
  in_diff->CopyDataContentFrom(ctx.device_ctx, bw_buf);
}

template<>
struct NcclInterDeviceReduceSumKernelUtil<DeviceType::kCPU> {
  static void ReduceSum(DeviceCtx* ctx, Blob* send, Blob* recv) { UNIMPLEMENTED(); }
};

}  // namespace oneflow
