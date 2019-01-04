#include "oneflow/core/kernel/nccl_inter_device_reduce_kernel.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace {

template<typename T>
void NcclInterDeviceReduce(DeviceCtx* ctx, const NcclInterDeviceReduceMethod method, Blob* send,
                           Blob* recv) {
  if (method == NcclInterDeviceReduceMethod::kSum) {
    NcclUtil::AllReduce(ctx, send, recv);
  } else if (method == NcclInterDeviceReduceMethod::kMean) {
    NcclUtil::AllReduce(ctx, send, recv);
    int32_t num_rank;
    NcclUtil::GetNumRanks(ctx, &num_rank);
    KernelUtil<DeviceType::kGPU, T>::MulByScalarPara(ctx, recv->shape().elem_cnt(), recv->dptr<T>(),
                                                     OneVal<T>::value / static_cast<T>(num_rank),
                                                     recv->mut_dptr<T>());
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void DoDataContent(DeviceCtx* ctx, const NcclInterDeviceReduceOpConf& conf, Blob* in, Blob* out,
                   Blob* buf) {
  if (buf) {
    Memcpy<DeviceType::kGPU>(ctx, buf->mut_dptr(), in->dptr(), in->ByteSizeOfDataContentField());
    NcclInterDeviceReduce<T>(ctx, conf.method(), buf, buf);
    Memcpy<DeviceType::kGPU>(ctx, out->mut_dptr(), buf->dptr(), out->ByteSizeOfDataContentField());
  } else {
    NcclInterDeviceReduce<T>(ctx, conf.method(), in, out);
  }
}

}  // namespace

template<typename T>
const PbMessage& NcclInterDeviceReduceKernel<T>::GetCustomizedOpConf() const {
  return this->op_conf().nccl_inter_device_reduce_conf();
}

template<typename T>
void NcclInterDeviceReduceKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DoDataContent<T>(ctx.device_ctx, op_conf().nccl_inter_device_reduce_conf(), BnInOp2Blob("in"),
                   BnInOp2Blob("out"), BnInOp2Blob("fw_buf"));
}

template<typename T>
void NcclInterDeviceReduceKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DoDataContent<T>(ctx.device_ctx, op_conf().nccl_inter_device_reduce_conf(),
                   BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob(GenDiffBn("in")),
                   BnInOp2Blob("bw_buf"));
}

namespace {

Kernel* CreateKernel(const KernelConf& kernel_conf) {
  static const HashMap<int, std::function<Kernel*()>> creators = {
#define MAKE_NCCL_INTER_DEVICE_REDUCE_KERNEL_CREATOR_ENTRY(cpp_type, data_type) \
  {data_type, []() { return new NcclInterDeviceReduceKernel<cpp_type>(); }},
      OF_PP_FOR_EACH_TUPLE(MAKE_NCCL_INTER_DEVICE_REDUCE_KERNEL_CREATOR_ENTRY,
                           FLOATING_DATA_TYPE_SEQ)};
#undef MAKE_NCCL_INTER_DEVICE_REDUCE_KERNEL_CREATOR_ENTRY
  return creators.at(kernel_conf.data_type())();
}

REGISTER_KERNEL_CREATOR(OperatorConf::kNcclInterDeviceReduceConf, CreateKernel);

}  // namespace

}  // namespace oneflow
