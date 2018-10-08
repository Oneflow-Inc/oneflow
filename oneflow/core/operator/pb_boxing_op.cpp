#include "oneflow/core/operator/pb_boxing_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void PbBoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  CHECK(boxing_conf().lbi().is_pb_blob());
  OpAttribute* op_attribute = kernel_conf->mutable_op_attribute();
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, op_attribute->mutable_pb_input_bns());
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, op_attribute->mutable_pb_output_bns());
}

void PbBoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_pb_boxing_conf());
  CHECK_NE(boxing_conf().in_box_case(), BoxingOpConf::kAddBox);

  for (int32_t i = 0; i < boxing_conf().in_num(); ++i) {
    EnrollPbInputBn("in_" + std::to_string(i));
  }
  for (int32_t i = 0; i < boxing_conf().out_num(); ++i) {
    EnrollPbOutputBn("out_" + std::to_string(i));
  }
}

const BoxingOpConf& PbBoxingOp::boxing_conf() const {
  return op_conf().pb_boxing_conf().boxing_conf();
}

void PbBoxingOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  std::vector<int64_t> instance_inner_shape_vec;
  std::vector<int64_t> fake_data_tmp_blob_shape_vec =
      CalcDataTmpBlobShapeVec(GetBlobDesc4BnInOp, &instance_inner_shape_vec);
  InferOutBlobDescs(GetBlobDesc4BnInOp, fake_data_tmp_blob_shape_vec, instance_inner_shape_vec);
}

REGISTER_OP(OperatorConf::kPbBoxingConf, PbBoxingOp);

}  // namespace oneflow
