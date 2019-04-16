#include "oneflow/core/operator/reduce_sum_op.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

void ReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().reduce_sum_conf().has_in_sys()) {
    EnrollDataTmpBn("fw_tmp");
  } else {
    EnrollFwBufBn("fw_tmp");
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

void ReduceSumOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  const auto& reduced_axes = op_conf().reduce_sum_conf().axis();
  ReduceSbpUtil::GetReduceSumSplitSignatureRules(this, "in",
                                                 {reduced_axes.begin(), reduced_axes.end()}, rules);
  rules->emplace_back(MakeSoleIbnBroadcastSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
