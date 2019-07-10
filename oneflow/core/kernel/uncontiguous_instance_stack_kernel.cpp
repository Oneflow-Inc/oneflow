#include "oneflow/core/kernel/uncontiguous_instance_stack_kernel.h"
#include "oneflow/core/kernel/piece_slice_v2_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void UncontiguousInstanceStackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<const Blob*> in_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().uncontiguous_instance_stack_conf().in_size()) {
    in_blobs.push_back(BnInOp2Blob("in_" + std::to_string(i)));
  }
  Blob* out_blob = BnInOp2Blob("out");
  PieceSliceV2KernelUtil<device_type, T>::InstanceStack(ctx.device_ctx, in_blobs, out_blob);
}

template<DeviceType device_type, typename T>
void UncontiguousInstanceStackKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  std::vector<Blob*> in_diff_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().uncontiguous_instance_stack_conf().in_size()) {
    in_diff_blobs.push_back(BnInOp2Blob(GenDiffBn("in_" + std::to_string(i))));
  }
  PieceSliceV2KernelUtil<device_type, T>::PieceSlice(ctx.device_ctx, out_diff_blob, in_diff_blobs);
}

template<DeviceType device_type, typename T>
void UncontiguousInstanceStackKernel<device_type, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(size_t, i, 0, this->op_conf().uncontiguous_instance_stack_conf().in_size()) {
    Blob* in_i_blob = BnInOp2Blob("in_" + std::to_string(i));
    CHECK(in_i_blob->has_dim0_valid_num_field());
    out_blob->set_dim1_valid_num(i, in_i_blob->shape().At(0));
  }
}

template<DeviceType device_type, typename T>
void UncontiguousInstanceStackKernel<device_type, T>::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(size_t, i, 0, this->op_conf().uncontiguous_instance_stack_conf().in_size()) {
    Blob* in_i_blob = BnInOp2Blob("in_" + std::to_string(i));
    CHECK(in_i_blob->has_dim1_valid_num_field());
    FOR_RANGE(size_t, j, 0, in_i_blob->shape().At(0)) {
      out_blob->set_dim2_valid_num(i, j, in_i_blob->dim1_valid_num(j));
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kUncontiguousInstanceStackConf,
                           UncontiguousInstanceStackKernel, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
