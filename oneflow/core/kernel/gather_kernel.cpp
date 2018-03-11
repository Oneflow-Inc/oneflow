#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Shape& in_shape = in_blob->shape();
  GatherKernelUtil<device_type, T>::Gather(
      ctx.device_ctx, in_shape.elem_cnt(),
      static_cast<int32_t>(in_shape.Count(1)),
      *static_cast<int32_t*>(ctx.other), in_blob->dptr<T>(), in_blob->col_num(),
      BnInOp2Blob("out")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (*static_cast<int32_t*>(ctx.other) == 0) {
    KernelIf<device_type>::ForwardDataId(ctx, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (*static_cast<int32_t*>(ctx.other) == 0) {
    KernelIf<device_type>::ForwardColNum(ctx, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Shape& out_diff_shape = out_diff_blob->shape();
  GatherKernelUtil<device_type, T>::Gather(
      ctx.device_ctx, out_diff_shape.elem_cnt(),
      static_cast<int32_t>(out_diff_shape.Count(1)),
      *static_cast<int32_t*>(ctx.other), out_diff_blob->dptr<T>(),
      out_diff_blob->col_num(), in_diff_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (*static_cast<int32_t*>(ctx.other) == 0) {
    KernelIf<device_type>::BackwardColNum(ctx, BnInOp2Blob);
  }
}

template<typename T>
class GatherKernelUtil<DeviceType::kCPU, T> {
 public:
  static void Gather(DeviceCtx* ctx, const int64_t n, const int32_t hidden_dim,
                     const int32_t col_id, const T* src_dptr,
                     const int32_t* col_num_ptr, T* dst_dptr) {
    // TODO
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
