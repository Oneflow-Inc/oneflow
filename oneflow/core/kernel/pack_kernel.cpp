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
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;
  size_t total_pack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(0, total_pack_num % out_blob->dim0_inner_shape().At(0));
  size_t pack_num4each_count1 = total_pack_num / out_blob->dim0_inner_shape().At(0);
  CHECK_EQ(0, out_blob->dim0_inner_shape().Count(1) % pack_num4each_count1);
  CHECK_EQ(2, in_blob->dim0_inner_shape().NumAxes());
  CHECK_EQ(1, in_blob->dim0_inner_shape().At(0));
  CHECK_EQ(in_blob->static_shape().At(0), in_blob->dim0_inner_shape().Count(1));

  if (in_index == 0) {
    for (size_t i = 0; i < out_blob->dim0_inner_shape().At(0); ++i) {
      out_blob->set_dim0_valid_num(i, 0);
    }
  }

  size_t part_size = out_blob->dim0_inner_shape().Count(1) / pack_num4each_count1;
  size_t idx_of_cur_valid_num = in_index / pack_num4each_count1;
  size_t idx_in_cur_valid_num = in_index % pack_num4each_count1;
  if (in_blob->dim0_valid_num(0) > 0) {
    CHECK_EQ(out_blob->dim0_valid_num(idx_of_cur_valid_num), idx_in_cur_valid_num * part_size);
    out_blob->set_dim0_valid_num(idx_of_cur_valid_num,
                                 idx_in_cur_valid_num * part_size + in_blob->dim0_valid_num(0));
  }
}

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  for (size_t i = 0; i < in_blob->shape().At(0); ++i) {
    out_blob->set_dim1_valid_num(i + in_index * in_blob->static_shape().At(0),
                                 in_blob->dim1_valid_num(i));
  }
}

template<DeviceType device_type>
void PackKernel<device_type>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  for (size_t i = 0; i < in_blob->shape().At(0); ++i) {
    out_blob->set_record_id_in_device_piece(i + in_index * in_blob->static_shape().At(0),
                                            in_blob->record_id_in_device_piece(i));
  }
}

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  for (size_t i = 0; i < in_blob->shape().At(0); ++i) {
    Memcpy<DeviceType::kCPU>(
        ctx.device_ctx, out_blob->mut_data_id(i + in_index * in_blob->static_shape().At(0)),
        in_blob->data_id(i), Global<JobDesc>::Get()->other_conf().max_data_id_length());
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kPackConf, PackKernel);

}  // namespace oneflow
