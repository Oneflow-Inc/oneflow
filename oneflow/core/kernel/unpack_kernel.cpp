#include "oneflow/core/kernel/unpack_kernel.h"
#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;
  PackKernelUtil<device_type>::Unpack(ctx.device_ctx, out_index, total_unpack_num,
                                      BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type>
void UnpackKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_diff_index = res->first;
  size_t total_pack_num = res->second;
  PackKernelUtil<device_type>::Pack(ctx.device_ctx, out_diff_index, total_pack_num,
                                    BnInOp2Blob("out_diff"), BnInOp2Blob("in_diff"));
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(0, in_blob->dim0_inner_shape().Count(1) % total_unpack_num);
  CHECK_EQ(2, out_blob->dim0_inner_shape().NumAxes());
  CHECK_EQ(1, out_blob->dim0_inner_shape().At(0));
  CHECK_EQ(out_blob->static_shape().At(0), out_blob->dim0_inner_shape().At(1));

  size_t total_num4each_count1 = in_blob->dim0_inner_shape().Count(1) / total_unpack_num;
  size_t valid_num_idx = out_index / total_unpack_num;
  size_t idx4cur_valid_num = out_index % total_unpack_num;
  size_t cur_valid_num = in_blob->dim0_valid_num(valid_num_idx);
  if ((idx4cur_valid_num + 1) * total_num4each_count1 <= cur_valid_num) {
    out_blob->set_dim0_valid_num(1, total_num4each_count1);
  } else if (idx4cur_valid_num * total_num4each_count1 >= cur_valid_num) {
    out_blob->set_dim0_valid_num(1, 0);
  } else {
    out_blob->set_dim0_valid_num(1, cur_valid_num - idx4cur_valid_num * total_num4each_count1);
  }
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t out_size = out_blob->static_shape().At(0);
  for (size_t i = 0; i < out_size; ++i) {
    out_blob->set_dim1_valid_num(i, in_blob->dim1_valid_num(i + out_index * out_size));
  }
}

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardRecordIdxInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t out_size = out_blob->static_shape().At(0);
  for (size_t i = 0; i < out_size; ++i) {
    out_blob->set_record_idx_in_device_piece(
        i, in_blob->record_idx_in_device_piece(i + out_index * out_size));
  }
}

template<DeviceType device_type>
void UnpackKernel<device_type>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("in_diff")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kUnpackConf, UnpackKernel);

}  // namespace oneflow
