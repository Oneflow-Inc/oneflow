#include "oneflow/core/kernel/piece_slice_v2_kernel.h"
#include "oneflow/core/kernel/piece_slice_v2_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  std::vector<Blob*> out_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().piece_slice_v2_conf().out_size()) {
    out_blobs.push_back(BnInOp2Blob("out_" + std::to_string(i)));
  }
  PieceSliceV2KernelUtil<device_type, T>::PieceSlice(ctx.device_ctx, in_blob, out_blobs);
}

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<const Blob*> out_diff_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().piece_slice_v2_conf().out_size()) {
    out_diff_blobs.push_back(BnInOp2Blob(GenDiffBn("out_" + std::to_string(i))));
  }
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  PieceSliceV2KernelUtil<device_type, T>::InstanceStack(ctx.device_ctx, out_diff_blobs,
                                                        in_diff_blob);
}

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const bool uncontiguous_varing_in =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_in = in_blob->has_instance_shape_field();
  FOR_RANGE(size_t, i, 0, this->op_conf().piece_slice_v2_conf().out_size()) {
    Blob* out_i_blob = BnInOp2Blob("out_" + std::to_string(i));
    if (uncontiguous_varing_in) {
      CHECK(!contiguous_varing_in);
      CHECK(in_blob->has_dim1_valid_num_field());
      out_i_blob->set_dim0_valid_num(0, in_blob->dim1_valid_num(i));
    } else if (contiguous_varing_in) {
      CHECK(!uncontiguous_varing_in);
      out_i_blob->set_dim0_valid_num(0, in_blob->shape().At(1));
    } else {
      UNIMPLEMENTED();
    }
  }
}

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  std::vector<Blob*> out_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().piece_slice_v2_conf().out_size()) {
    out_blobs.push_back(BnInOp2Blob("out_" + std::to_string(i)));
  }
  PieceSliceV2KernelUtil<device_type, T>::SliceInstanceShape(in_blob, out_blobs);
}

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const bool uncontiguous_varing_in =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_in = in_blob->has_instance_shape_field();
  FOR_RANGE(size_t, i, 0, this->op_conf().piece_slice_v2_conf().out_size()) {
    Blob* out_i_blob = BnInOp2Blob("out_" + std::to_string(i));
    if (uncontiguous_varing_in) {
      CHECK(!contiguous_varing_in);
      FOR_RANGE(size_t, j, 0, in_blob->dim1_valid_num(i)) {
        out_i_blob->set_dim1_valid_num(j, in_blob->dim2_valid_num(i, j));
      }
    } else {
      UNIMPLEMENTED();
    }
  }
}

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<DeviceType device_type, typename T>
void PieceSliceV2Kernel<device_type, T>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<const Blob*> out_diff_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().piece_slice_v2_conf().out_size()) {
    out_diff_blobs.push_back(BnInOp2Blob(GenDiffBn("out_" + std::to_string(i))));
  }
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  PieceSliceV2KernelUtil<device_type, T>::StackInstanceShape(out_diff_blobs, in_diff_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPieceSliceV2Conf, PieceSliceV2Kernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
