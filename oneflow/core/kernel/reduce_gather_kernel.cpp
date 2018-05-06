#include "oneflow/core/kernel/reduce_gather_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceGatherKernel<device_type>::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_id_ = ctx->parallel_id();
}

template<DeviceType device_type>
void ReduceGatherKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  char* dst_cur_dptr = BnInOp2Blob("out")->mut_dptr<char>();
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  for (int32_t i = 0; i < input_bns.size(); ++i) {
    Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
    ReduceGatherKernelUtil<device_type>::DoMemcpy(
        ctx.device_ctx, dst_cur_dptr, in_blob->dptr<char>(), in_byte_size, i == parallel_id_);
    dst_cur_dptr += in_byte_size;
  }
}

template<>
struct ReduceGatherKernelUtil<DeviceType::kCPU> {
  static void DoMemcpy(DeviceCtx* ctx, char* dst, const char* src, size_t sz,
                       bool is_same_parallel_id) {
    Memcpy<DeviceType::kCPU>(ctx, dst, src, sz);
  }
};

template<>
struct ReduceGatherKernelUtil<DeviceType::kGPU> {
  static void DoMemcpy(DeviceCtx* ctx, char* dst, const char* src, size_t sz,
                       bool is_same_parallel_id) {
    if (is_same_parallel_id) {
      Memcpy<DeviceType::kGPU>(ctx, dst, src, sz);
    } else {
      Memcpy<DeviceType::kGPU>(ctx, dst, src, sz, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }
  }
};

namespace {

Kernel* CreateReduceGatherKernel(const KernelConf& kernel_conf) {
  static const HashMap<int32_t, std::function<Kernel*()>> creators = {
#define REDUCE_GATHER_KERNEL_ENTRY(device_type) \
  {device_type, []() { return new ReduceGatherKernel<device_type>; }},
      OF_PP_FOR_EACH_TUPLE(REDUCE_GATHER_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(kernel_conf.op_attribute().device_type())();
}

}  // namespace

#define INSTANTIATE_REDUCE_GATHER_KERNEL_UTIL(device_type) \
  template struct ReduceGatherKernelUtil<device_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_REDUCE_GATHER_KERNEL_UTIL, DEVICE_TYPE_SEQ)

REGISTER_KERNEL_CREATOR(OperatorConf::kReduceGatherConf, CreateReduceGatherKernel);

}  // namespace oneflow
