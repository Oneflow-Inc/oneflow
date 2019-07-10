#include "oneflow/core/kernel/contiguous_instance_stack_kernel.h"
#include "oneflow/core/kernel/piece_slice_v2_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ContiguousInstanceStackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<const Blob*> in_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().contiguous_instance_stack_conf().in_size()) {
    in_blobs.push_back(BnInOp2Blob("in_" + std::to_string(i)));
  }
  Blob* out_blob = BnInOp2Blob("out");
  PieceSliceV2KernelUtil<device_type, T>::InstanceStack(ctx.device_ctx, in_blobs, out_blob);
}

template<DeviceType device_type, typename T>
void ContiguousInstanceStackKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  std::vector<Blob*> in_diff_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().contiguous_instance_stack_conf().in_size()) {
    in_diff_blobs.push_back(BnInOp2Blob(GenDiffBn("in_" + std::to_string(i))));
  }
  PieceSliceV2KernelUtil<device_type, T>::PieceSlice(ctx.device_ctx, out_diff_blob, in_diff_blobs);
}

template<DeviceType device_type, typename T>
void ContiguousInstanceStackKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<DeviceType device_type, typename T>
void ContiguousInstanceStackKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<const Blob*> in_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().contiguous_instance_stack_conf().in_size()) {
    in_blobs.push_back(BnInOp2Blob("in_" + std::to_string(i)));
  }
  Blob* out_blob = BnInOp2Blob("out");
  PieceSliceV2KernelUtil<device_type, T>::StackInstanceShape(in_blobs, out_blob);
}

template<DeviceType device_type, typename T>
void ContiguousInstanceStackKernel<device_type, T>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  CHECK(!out_diff_blob->has_dim1_valid_num_field());
  FOR_RANGE(size_t, i, 0, this->op_conf().contiguous_instance_stack_conf().in_size()) {
    Blob* in_i_diff_blob = BnInOp2Blob(GenDiffBn("in_" + std::to_string(i)));
    CHECK(in_i_diff_blob->has_dim0_valid_num_field());
    in_i_diff_blob->set_dim0_valid_num(0, out_diff_blob->shape().At(1));
  }
}

template<DeviceType device_type, typename T>
void ContiguousInstanceStackKernel<device_type, T>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  std::vector<Blob*> in_diff_blobs;
  FOR_RANGE(size_t, i, 0, this->op_conf().contiguous_instance_stack_conf().in_size()) {
    in_diff_blobs.push_back(BnInOp2Blob(GenDiffBn("in_" + std::to_string(i))));
  }
  PieceSliceV2KernelUtil<device_type, T>::SliceInstanceShape(out_diff_blob, in_diff_blobs);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kContiguousInstanceStackConf,
                           ContiguousInstanceStackKernel, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
