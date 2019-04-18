#include "oneflow/core/operator/reduce_mean_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/sbp_signature_rule.h"

namespace oneflow {

namespace {

class ReduceMeanBroadcastSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMeanBroadcastSignatureRule);
  ~ReduceMeanBroadcastSignatureRule() override = default;

  ReduceMeanBroadcastSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (B,) -> (B,)"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["in"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_broadcast_parallel();
  }
};

}  // namespace

void ReduceMeanOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_mean_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_tmp");
  EnrollBwBufBn("bw_tmp");
}

const PbMessage& ReduceMeanOp::GetCustomizedConf() const { return op_conf().reduce_mean_conf(); }

void ReduceMeanOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext*) const {
  const ReduceMeanOpConf& conf = op_conf().reduce_mean_conf();
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

void ReduceMeanOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("bw_tmp") = *GetBlobDesc4BnInOp("out");
}

void ReduceMeanOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  const auto& reduced_axes = op_conf().reduce_mean_conf().axis();
  HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  if (ReduceSbpUtil::IsReduceAxisSplitted(SbpInferHint4Ibn("in"), conf_axes) == false) {
    rules->emplace_back(MakeDataSplitSbpSignatureRule(this));
  } else {
    rules->emplace_back(new ReduceMeanBroadcastSignatureRule(this));
  }
}

REGISTER_OP(OperatorConf::kReduceMeanConf, ReduceMeanOp);

}  // namespace oneflow
