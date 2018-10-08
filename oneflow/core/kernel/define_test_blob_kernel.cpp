#include "oneflow/core/kernel/define_test_blob_kernel.h"

namespace oneflow {

void DefineTestBlobKernel::ForwardInstanceVaryingElemCnt(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = op_conf().define_test_blob_conf();
  if (!conf.has_instance_varying_elem_cnt()) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int32_t, i, 0, out_blob->shape().At(0)) {
    out_blob->set_instance_varying_elem_cnt(i, conf.instance_varying_elem_cnt());
  }
}

void DefineTestBlobKernel::ForwardVaryingInstanceNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = op_conf().define_test_blob_conf();
  if (!conf.has_varying_instance_num()) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int32_t, i, 0, out_blob->instance_inner_shape()->At(0)) {
    out_blob->set_varying_instance_num(i, conf.varying_instance_num());
  }
}

REGISTER_KERNEL(OperatorConf::kDefineTestBlobConf, DefineTestBlobKernel);

}  // namespace oneflow
