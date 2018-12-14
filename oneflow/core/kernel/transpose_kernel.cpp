#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Transpose<device_type, T>(ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("out"),
                            this->kernel_conf().transpose_conf().perm());
}

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff = BnInOp2Blob("in_diff");
  if (in_diff) {
    Transpose<device_type, T>(ctx.device_ctx, BnInOp2Blob("out_diff"), in_diff,
                              this->kernel_conf().transpose_conf().invert_perm());
  }
}

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Shape& in_shape = BnInOp2Blob("in")->shape();
  const PbRf<int32_t>& perm = this->kernel_conf().transpose_conf().perm();
  std::vector<int64_t> dim_vec(perm.size() - 1, -1);
  FOR_RANGE(size_t, i, 1, perm.size()) { dim_vec[i - 1] = in_shape.At(perm[i]); }
  BnInOp2Blob("out")->set_instance_shape(Shape(dim_vec));
}

template<DeviceType device_type, typename T>
void TransposeKernel<device_type, T>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (!in_diff_blob) { return; }
  const Shape& out_diff_shape = BnInOp2Blob("out_diff")->shape();
  const PbRf<int32_t>& invert_perm = this->kernel_conf().transpose_conf().invert_perm();
  std::vector<int64_t> dim_vec(invert_perm.size() - 1, -1);
  FOR_RANGE(size_t, i, 1, invert_perm.size()) {
    dim_vec[i - 1] = out_diff_shape.At(invert_perm[i]);
  }
  in_diff_blob->set_instance_shape(Shape(dim_vec));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTransposeConf, TransposeKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
