#include "oneflow/core/kernel/nccl_inter_device_reduce_kernel.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace {

void NcclInterDeviceReduceSum(DeviceCtx* ctx, Blob* send, Blob* recv) {
  NcclUtil::AllReduce(ctx, send, recv);
}

}  // namespace

void NcclInterDeviceReduceKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const bool use_buf = BnInOp2Blob("fw_buf") != nullptr;
  if (use_buf) {
    Blob* fw_buf = BnInOp2Blob("fw_buf");
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, fw_buf->mut_dptr(), in->dptr(),
                             in->ByteSizeOfDataContentField());
    NcclInterDeviceReduceSum(ctx.device_ctx, fw_buf, fw_buf);
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, out->mut_dptr(), fw_buf->dptr(),
                             out->ByteSizeOfDataContentField());
  } else {
    NcclInterDeviceReduceSum(ctx.device_ctx, in, out);
  }
}

void NcclInterDeviceReduceKernel::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  const bool use_buf = BnInOp2Blob("bw_buf") != nullptr;
  if (use_buf) {
    Blob* bw_buf = BnInOp2Blob("bw_buf");
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, bw_buf->mut_dptr(), out_diff->dptr(),
                             out_diff->ByteSizeOfDataContentField());
    NcclInterDeviceReduceSum(ctx.device_ctx, bw_buf, bw_buf);
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, in_diff->mut_dptr(), bw_buf->dptr(),
                             in_diff->ByteSizeOfDataContentField());
  } else {
    NcclInterDeviceReduceSum(ctx.device_ctx, out_diff, in_diff);
  }
}

REGISTER_KERNEL(OperatorConf::kNcclInterDeviceReduceConf, NcclInterDeviceReduceKernel);

}  // namespace oneflow
