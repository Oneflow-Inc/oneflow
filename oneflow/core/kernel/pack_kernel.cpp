#include "oneflow/core/kernel/pack_kernel.h"
#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;
  size_t total_pack_num = res->second;
  PackKernelUtil<device_type>::Pack(ctx.device_ctx, in_index, total_pack_num, BnInOp2Blob("in"),
                                    BnInOp2Blob("out"));
}

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;
  size_t total_pack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t in_size = in_blob->static_shape().At(0);
  for (size_t i = 0; i < in_size; ++i) {
    out_blob->set_dim1_valid_num(i + in_index * in_size, in_blob->dim1_valid_num(i));
  }
}

template<DeviceType device_type>
void PackKernel<device_type>::ForwardRecordIdxInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;
  size_t total_pack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t in_size = in_blob->static_shape().At(0);
  for (size_t i = 0; i < in_size; ++i) {
    out_blob->set_record_idx_in_device_piece(i + in_index * in_size,
                                             in_blob->record_idx_in_device_piece(i));
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kPackConf, PackKernel);

}  // namespace oneflow
