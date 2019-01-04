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

}  // namespace

template<typename T>
void NcclInterDeviceReduceKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NcclInterDeviceReduceOpConf& conf = op_conf().nccl_inter_device_reduce_conf();
  Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const bool use_buf = BnInOp2Blob("fw_buf") != nullptr;
  if (use_buf) {
    Blob* fw_buf = BnInOp2Blob("fw_buf");
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, fw_buf->mut_dptr(), in->dptr(),
                             in->ByteSizeOfDataContentField());
    NcclInterDeviceReduce<T>(ctx.device_ctx, conf.method(), fw_buf, fw_buf);
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, out->mut_dptr(), fw_buf->dptr(),
                             out->ByteSizeOfDataContentField());
  } else {
    NcclInterDeviceReduce<T>(ctx.device_ctx, conf.method(), in, out);
  }
}

template<typename T>
void NcclInterDeviceReduceKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NcclInterDeviceReduceOpConf& conf = op_conf().nccl_inter_device_reduce_conf();
  Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  const bool use_buf = BnInOp2Blob("bw_buf") != nullptr;
  if (use_buf) {
    Blob* bw_buf = BnInOp2Blob("bw_buf");
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, bw_buf->mut_dptr(), out_diff->dptr(),
                             out_diff->ByteSizeOfDataContentField());
    NcclInterDeviceReduce<T>(ctx.device_ctx, conf.method(), bw_buf, bw_buf);
    Memcpy<DeviceType::kGPU>(ctx.device_ctx, in_diff->mut_dptr(), bw_buf->dptr(),
                             in_diff->ByteSizeOfDataContentField());
  } else {
    NcclInterDeviceReduce<T>(ctx.device_ctx, conf.method(), out_diff, in_diff);
  }
}

#define MAKE_GPU_KERNEL_CREATOR_ENTRY(kernel_class, data_type_pair) \
  {OF_PP_PAIR_SECOND(data_type_pair),                               \
   []() { return new kernel_class<OF_PP_PAIR_FIRST(data_type_pair)>(); }},

#define ADD_GPU_DEFAULT_KERNEL_CREATOR(op_type_case, kernel_class, data_type_seq)       \
  namespace {                                                                           \
                                                                                        \
  Kernel* CreateKernel(const KernelConf& kernel_conf) {                                 \
    static const HashMap<int, std::function<Kernel*()>> creators = {                    \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_GPU_KERNEL_CREATOR_ENTRY, (kernel_class), \
                                         data_type_seq)};                               \
    return creators.at(kernel_conf.data_type())();                                      \
  }                                                                                     \
                                                                                        \
  REGISTER_KERNEL_CREATOR(op_type_case, CreateKernel);                                  \
  }

ADD_GPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLayerNormConf, NcclInterDeviceReduceKernel,
                               FLOATING_DATA_TYPE_SEQ);
#undef ADD_GPU_DEFAULT_KERNEL_CREATOR
#undef MAKE_GPU_KERNEL_CREATOR_ENTRY

}  // namespace oneflow
