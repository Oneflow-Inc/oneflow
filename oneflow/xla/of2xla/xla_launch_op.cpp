#include "oneflow/xla/of2xla/xla_launch_op.h"
#include "absl/strings/str_split.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_launch_attr.h"
#include "oneflow/xla/of2xla/xla_node.h"
#include "oneflow/xla/of2xla/xla_utility.h"

namespace oneflow {

void XlaLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xla_launch_conf());
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  mola::LaunchAttrHelper attr_helper(xla_launch_conf.attr());
  int inputs_num = xla_launch_conf.in().size();
  int outputs_num = xla_launch_conf.out().size();
  for (int i = 0; i < inputs_num; ++i) {
    const std::string &input = xla_launch_conf.in().at(i);
    // bool is_mutable = true;
    bool is_mutable = attr_helper.IsMutableArg(input);
    EnrollInputBn(absl::StrCat("in_", i))->set_is_mutable(is_mutable);
  }
  if (outputs_num > 0) {
    EnrollRepeatedOutputBn("out");
  }
  // Setup subgraph
  subgraph_.reset(new mola::XlaLaunchGraph(xla_launch_conf, &this->job_desc()));
}

const PbMessage &XlaLaunchOp::GetCustomizedConf() const {
  return op_conf().xla_launch_conf();
}

Maybe<void> XlaLaunchOp::InferBlobDescs(
    std::function<BlobDesc *(const std::string &)> GetBlobDesc4BnInOp,
    const ParallelContext *parallel_ctx) const {
  // Prepare outer input blob descs
  std::unordered_map<std::string, BlobDesc> blob_descs;
  for (const std::string &bn : this->input_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = BlobName(lbi);
    blob_descs[blob_name].CopyMetaFrom(*GetBlobDesc4BnInOp(bn));
  }

  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  SbpSignature sbp_signature = RestoreSbpSignature(xla_launch_conf);
  // Inference blob descs in subgraph
  subgraph_->InferBlobDescs(&blob_descs, *parallel_ctx, sbp_signature);

  // Fetch output blob descs
  for (const std::string &bn : this->output_bns()) {
    CHECK_GT(blob_descs.count(bn), 0);
    *GetBlobDesc4BnInOp(bn) = blob_descs[bn];
  }
  return Maybe<void>::Ok();
}

Maybe<void> XlaLaunchOp::InferBatchAxis(
    std::function<OptInt64 *(const std::string &)> BatchAxis4BnInOp) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  const auto &resource_scope = xla_launch_conf.attr().resource_scope();
  const auto &batch_axis = resource_scope.batch_axis();

  for (const std::string &bn : this->output_bns()) {
    LogicalBlobId lbi = subgraph_->Output(bn);
    std::string blob_name = BlobName(lbi);
    CHECK_GT(batch_axis.count(blob_name), 0);
    *BatchAxis4BnInOp(bn) = batch_axis.at(blob_name);
  }
  return Maybe<void>::Ok();
}

Maybe<void> XlaLaunchOp::InferSbpSignature(
    SbpSignature *sbp_signature, const SbpSignature &sbp_sig_conf,
    const std::function<int32_t(const SbpSignature &)> &CalcOrderValue4SbpSig,
    XlaLaunchOp::SbpInferHint4IbnFunc SbpInferHint4Ibn,
    const ParallelDesc &parallel_desc) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  *sbp_signature = RestoreSbpSignature(xla_launch_conf);

  const auto &bn2sbp_parallel = sbp_signature->bn_in_op2sbp_parallel();
  for (const std::string &bn : this->input_bns()) {
    CHECK_GT(bn2sbp_parallel.count(bn), 0);
  }
  for (const std::string &bn : this->output_bns()) {
    CHECK_GT(bn2sbp_parallel.count(bn), 0);
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kXlaLaunchConf, XlaLaunchOp);

}  // namespace oneflow
