#include "oneflow/core/kernel/reduce_scatter_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceScatterKernel<device_type>::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_id_ = ctx->parallel_id();
}

template<DeviceType device_type>
void ReduceScatterKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const char* src_cur_dptr = BnInOp2Blob("in")->dptr<char>();
  const PbRpf<std::string>& output_bns = this->op_attribute().output_bns();
  for (int32_t i = 0; i < output_bns.size(); ++i) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
    ReduceScatterKernelUtil<device_type>::DoMemcpy(ctx.device_ctx, out_blob->mut_dptr<char>(),
                                                   src_cur_dptr, out_byte_size, i == parallel_id_);
    src_cur_dptr += out_byte_size;
  }
}

template<>
struct ReduceScatterKernelUtil<DeviceType::kCPU> {
  static void DoMemcpy(DeviceCtx* ctx, char* dst, const char* src, size_t sz,
                       bool is_same_parallel_id) {
    Memcpy<DeviceType::kCPU>(ctx, dst, src, sz);
  }
};

template<>
struct ReduceScatterKernelUtil<DeviceType::kGPU> {
  static void DoMemcpy(DeviceCtx* ctx, char* dst, const char* src, size_t sz,
                       bool is_same_parallel_id) {
    if (is_same_parallel_id) {
      Memcpy<DeviceType::kGPU>(ctx, dst, src, sz);
    } else {
      Memcpy<DeviceType::kGPU>(ctx, dst, src, sz, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }
  }
};

namespace {

Kernel* CreateReduceScatterKernel(const KernelConf& kernel_conf) {
  static const HashMap<int32_t, std::function<Kernel*()>> creators = {
#define REDUCE_SCATTER_KERNEL_ENTRY(device_type) \
  {device_type, []() { return new ReduceScatterKernel<device_type>; }},
      OF_PP_FOR_EACH_TUPLE(REDUCE_SCATTER_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(kernel_conf.op_attribute().device_type())();
}

}  // namespace

#define INSTANTIATE_REDUCE_SCATTER_KERNEL_UTIL(device_type) \
  template struct ReduceScatterKernelUtil<device_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_REDUCE_SCATTER_KERNEL_UTIL, DEVICE_TYPE_SEQ)

REGISTER_KERNEL_CREATOR(OperatorConf::kReduceScatterConf, CreateReduceScatterKernel);

}  // namespace oneflow
