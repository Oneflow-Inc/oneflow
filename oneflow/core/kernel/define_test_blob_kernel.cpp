#include "oneflow/core/kernel/define_test_blob_kernel.h"

namespace oneflow {

void DefineTestBlobKernel::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = op_conf().define_test_blob_conf();
  if (!conf.has_dim0_valid_num()) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, out_blob->dim0_inner_shape().At(0)) {
    out_blob->set_dim0_valid_num(i, conf.dim0_valid_num());
  }
}

void DefineTestBlobKernel::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = op_conf().define_test_blob_conf();
  if (!conf.has_dim1_valid_num()) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, out_blob->shape().At(0)) {
    out_blob->set_dim1_valid_num(i, conf.dim1_valid_num());
  }
}

void DefineTestBlobKernel::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = op_conf().define_test_blob_conf();
  if (!conf.has_dim2_valid_num()) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, out_blob->shape().At(0)) {
    FOR_RANGE(int64_t, j, 0, out_blob->shape().At(1)) {
      out_blob->set_dim2_valid_num(i, j, conf.dim2_valid_num());
    }
  }
}

void DefineTestBlobKernel::ForwardRecordIdxInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = op_conf().define_test_blob_conf();
  if (conf.record_idx_in_device_piece().empty()) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, out_blob->shape().At(0)) {
    out_blob->set_record_idx_in_device_piece(
        i, conf.record_idx_in_device_piece(i % conf.record_idx_in_device_piece_size()));
  }
}

REGISTER_KERNEL(OperatorConf::kDefineTestBlobConf, DefineTestBlobKernel);

}  // namespace oneflow
