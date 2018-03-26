#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  int32_t col_id = *(static_cast<int32_t*>(ctx.other));
  int32_t data_num = in_blob->shape().At(0);
  int32_t hid_dim = in_blob->shape().Count(1);
  CHECK_EQ(hid_dim, out_blob->shape().Count(1));

  for (int32_t i = 0; i < data_num; ++i) {
    if (in_blob->col_num(i) == col_id) {
      Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>() + i * hid_dim,
                          in_blob->dptr<T>() + i * hid_dim, hid_dim);
    }
  }
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
  int32_t col_id = *(static_cast<int32_t*>(ctx.other));
  int32_t data_num = out_diff_blob->shape().At(0);
  int32_t hid_dim = out_diff_blob->shape().Count(1);
  CHECK_EQ(hid_dim, in_diff_blob->shape().Count(1));

  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  for (int32_t i = 0; i < data_num; ++i) {
    if (out_diff_blob->col_num(i) == col_id) {
      Memcpy<device_type>(ctx.device_ctx,
                          in_diff_blob->mut_dptr<T>() + i * hid_dim,
                          out_diff_blob->dptr<T>() + i * hid_dim, hid_dim);
    }
  }
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelIf<device_type>::BackwardColNum(ctx, BnInOp2Blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
