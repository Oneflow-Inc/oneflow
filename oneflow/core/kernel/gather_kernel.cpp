#include "oneflow/core/kernel/gather_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_conf();
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices_blob = BnInOp2Blob("indices");
  const int64_t num_indices = indices_blob->shape().elem_cnt();
  const Blob* in_blob = BnInOp2Blob("in");
  const int64_t in_rows = in_blob->shape().At(0);
  const int64_t in_cols = in_blob->shape().Count(1);
  Blob* out = BnInOp2Blob("out");
  LookUpKernelUtil<device_type, T>::Forward(ctx.device_ctx, indices_blob->dptr<int32_t>(),
                                            num_indices, in_blob->dptr<T>(), in_rows, in_cols,
                                            out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

template<typename T>
struct LookUpKernelUtil<DeviceType::kCPU, T> final {
  static void Forward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices, const T* in,
                      int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_rows, int64_t in_cols, T* in_diff);
};

template<typename T>
void LookUpKernelUtil<DeviceType::kCPU, T>::Forward(DeviceCtx* ctx, const int32_t* indices,
                                                    int64_t num_indices, const T* in,
                                                    int64_t in_rows, int64_t in_cols, T* out) {}

template<typename T>
void LookUpKernelUtil<DeviceType::kCPU, T>::Backward(DeviceCtx* ctx, const int32_t* indices,
                                                     int64_t num_indices, const T* out_diff,
                                                     int64_t in_rows, int64_t in_cols, T* in_diff) {

}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
