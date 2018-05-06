#include "oneflow/core/kernel/reduce_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_id_ = ctx->parallel_id();
}

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  Blob* same_parallel_in_blob = BnInOp2Blob(input_bns.Get(parallel_id_));
  Blob* out_blob = BnInOp2Blob("out");
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(),
                      same_parallel_in_blob->dptr<char>(), out_blob->ByteSizeOfDataContentField());
  for (int32_t i = 0; i < input_bns.size(); ++i) {
    if (i == parallel_id_) { continue; }
    ReduceAddKernelUtil<device_type, T>::DoAdd(
        ctx.device_ctx, out_blob->mut_dptr<T>(), BnInOp2Blob(input_bns[parallel_id_])->dptr<T>(),
        out_blob->shape().elem_cnt(), same_parallel_in_blob->mut_dptr<T>());
  }
}

template<typename T>
struct ReduceAddKernelUtil<DeviceType::kCPU, T> {
  static void DoAdd(DeviceCtx* ctx, T* dst, const T* src, size_t n, T* tmp) {
    KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, n, 1.0, src, 1, dst, 1);
  }
};

template<typename T>
struct ReduceAddKernelUtil<DeviceType::kGPU, T> {
  static void DoAdd(DeviceCtx* ctx, T* dst, const T* src, size_t n, T* tmp) {
    Memcpy<DeviceType::kGPU>(ctx, tmp, src, n, cudaMemcpyKind::cudaMemcpyHostToDevice);
    KernelUtil<DeviceType::kGPU, T>::Axpy(ctx, n, 1.0, tmp, 1, dst, 1);
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceAddConf, ReduceAddKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
