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
  int64_t data_num = in_blob->shape().At(0);
  int64_t hid_dim = in_blob->shape().Count(1);
  CHECK_EQ(hid_dim, out_blob->shape().Count(1));

  for (int64_t i = 0; i < data_num; ++i) {
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
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  int32_t col_id = *(static_cast<int32_t*>(ctx.other));
  int64_t data_num = out_diff_blob->shape().At(0);
  int64_t hid_dim = out_diff_blob->shape().Count(1);
  CHECK_EQ(hid_dim, in_diff_blob->shape().Count(1));

  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  for (int64_t i = 0; i < data_num; ++i) {
    if (in_blob->col_num(i) == col_id) {
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
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  in_diff_blob->CopyColNumFrom<device_type>(ctx.device_ctx, in_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
