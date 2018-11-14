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
  LookupKernelUtil<device_type, T>::Forward(ctx.device_ctx, indices_blob->dptr<int32_t>(),
                                            num_indices, in_blob->dptr<T>(), in_rows, in_cols,
                                            out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices_blob = BnInOp2Blob("indices");
  const int64_t num_indices = indices_blob->shape().elem_cnt();
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  const int64_t in_rows = in_diff_blob->shape().At(0);
  const int64_t in_cols = in_diff_blob->shape().Count(1);
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  LookupKernelUtil<device_type, T>::Backward(ctx.device_ctx, indices_blob->dptr<int32_t>(),
                                             num_indices, out_diff_blob->dptr<T>(), in_rows,
                                             in_cols, in_diff_blob->mut_dptr<T>());
}

template<typename T>
struct LookupKernelUtil<DeviceType::kCPU, T> final {
  static void Forward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices, const T* in,
                      int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_rows, int64_t in_cols, T* in_diff);
};

template<typename T>
void LookupKernelUtil<DeviceType::kCPU, T>::Forward(DeviceCtx* ctx, const int32_t* indices,
                                                    int64_t num_indices, const T* in,
                                                    int64_t in_rows, int64_t in_cols, T* out) {
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const int64_t idx = indices[i];
    CHECK(idx >= 0 && idx < in_rows);
    const T* from = in + (idx * in_cols);
    T* to = out + (i * in_cols);
    std::copy(from, from + in_cols, to);
  }
}

template<typename T>
void LookupKernelUtil<DeviceType::kCPU, T>::Backward(DeviceCtx* ctx, const int32_t* indices,
                                                     int64_t num_indices, const T* out_diff,
                                                     int64_t in_rows, int64_t in_cols, T* in_diff) {
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const int64_t idx = indices[i];
    CHECK(idx >= 0 && idx < in_rows);
    const T* from = out_diff + (i * in_cols);
    T* to = in_diff + (idx * in_cols);
    std::transform(from, from + in_cols, to, to, std::plus<T>());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
