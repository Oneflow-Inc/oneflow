#include "oneflow/core/operator/reduce_sum_op.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().reduce_sum_conf().has_in_sys()) {
    EnrollTmpBn("fw_tmp");
  } else {
    EnrollTmpBn("fw_tmp");
  }
}

const PbMessage& ReduceSumOp::GetCustomizedConf() const { return op_conf().reduce_sum_conf(); }

void ReduceSumOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  if (conf.axis().empty()) {
    if (conf.keep_dims()) {
      out_blob->mut_shape() = Shape::Ones(in_blob->shape().NumAxes());
    } else {
      out_blob->mut_shape() = Shape({1});
    }
  } else {
    const std::vector<int64_t> axis_vec = {conf.axis().begin(), conf.axis().end()};
    const Shape& reduced_shape = in_blob->shape().CreateReducedShape(axis_vec);
    if (conf.keep_dims()) {
      out_blob->mut_shape() = reduced_shape;
    } else {
      out_blob->mut_shape() = reduced_shape.RemoveOnes(axis_vec);
    }
  }
}

void ReduceSumOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  const auto& reduced_axes = op_conf().reduce_sum_conf().axis();
  HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  *HasBatchDim4BnInOp("out") = !(conf_axes.empty() || (conf_axes.find(0) != conf_axes.end()));
}

void ReduceSumOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int32_t num_axes = LogicalBlobDesc4Ibn("in").shape().NumAxes();
  auto IsReducedAxis =
      ReduceSbpUtil::MakePredicatorIsReducedAxis(op_conf().reduce_sum_conf().axis(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .PartialSum(output_bns())
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    } else {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .Split(output_bns(), i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  }
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
