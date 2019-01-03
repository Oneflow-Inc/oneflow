#include "oneflow/core/kernel/nccl_inter_device_reduce_sum_kernel.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace {

void NcclInterDeviceReduceSum(DeviceCtx* ctx, Blob* send, Blob* recv) {
  NcclUtil::AllReduce(ctx, send, recv);
}

}  // namespace

void NcclInterDeviceReduceSumKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* fw_buf = BnInOp2Blob("fw_buf");
  Blob* out = BnInOp2Blob("out");
  Memcpy<DeviceType::kGPU>(ctx.device_ctx, fw_buf->mut_dptr(), in->dptr(),
                           in->ByteSizeOfDataContentField());
  NcclInterDeviceReduceSum(ctx.device_ctx, fw_buf, fw_buf);
  Memcpy<DeviceType::kGPU>(ctx.device_ctx, out->mut_dptr(), fw_buf->dptr(),
                           out->ByteSizeOfDataContentField());
}

void NcclInterDeviceReduceSumKernel::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* bw_buf = BnInOp2Blob("bw_buf");
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  Memcpy<DeviceType::kGPU>(ctx.device_ctx, bw_buf->mut_dptr(), out_diff->dptr(),
                           out_diff->ByteSizeOfDataContentField());
  NcclInterDeviceReduceSum(ctx.device_ctx, bw_buf, bw_buf);
  Memcpy<DeviceType::kGPU>(ctx.device_ctx, in_diff->mut_dptr(), bw_buf->dptr(),
                           in_diff->ByteSizeOfDataContentField());
}

REGISTER_KERNEL(OperatorConf::kNcclInterDeviceReduceSumConf, NcclInterDeviceReduceSumKernel);

}  // namespace oneflow
