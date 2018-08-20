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
  const BoxingOpConf& conf = boxing_conf();
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(pb_input_bns().Get(0));
  std::vector<int64_t> fake_data_tmp_blob_shape_vec =
      CalcDataTmpBlobShapeVec(GetBlobDesc4BnInOp, pb_input_bns());
  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), fake_data_tmp_blob_shape_vec.size());
    FOR_RANGE(size_t, i, 0, pb_output_bns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(pb_output_bns().Get(i));
      *out_blob_desc = *first_in_blob;
      fake_data_tmp_blob_shape_vec[split_conf.axis()] = split_conf.part_num(i);
      out_blob_desc->mut_shape() = Shape(fake_data_tmp_blob_shape_vec);
    }
  } else if (conf.out_box_case() == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : pb_output_bns()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(obn);
      *out_blob_desc = *first_in_blob;
      out_blob_desc->mut_shape() = Shape(fake_data_tmp_blob_shape_vec);
    }
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kPbBoxingConf, PbBoxingOp);

}  // namespace oneflow
