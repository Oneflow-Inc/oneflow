#include "absl/strings/str_split.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/xla_node.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_launch_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void XlaLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xla_launch_conf());

  int inputs_num = op_conf().xla_launch_conf().in().size();
  int outputs_num = op_conf().xla_launch_conf().out().size();
  for (int i = 0; i < inputs_num; ++i) {
    EnrollInputBn(absl::StrCat("in_", i))->set_is_mutable(true);
    // EnrollInputBn(absl::StrCat("in_", i));
  }
  if (outputs_num > 0) {
    EnrollRepeatedOutputBn("out");
  }
  // Setup subgraph
  subgraph_.reset(new mola::XlaLaunchGraph(
      op_conf().xla_launch_conf(), &this->job_desc()));
}

const PbMessage &XlaLaunchOp::GetCustomizedConf() const {
  return op_conf().xla_launch_conf();
}

Maybe<void> XlaLaunchOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  const auto &resource_scope = xla_launch_conf.attr().resource_scope();
  const auto &shapes = resource_scope.shapes();
  // Fetch output blob descs
  for (const std::string &bn : this->output_bns()) {
    LogicalBlobId lbi = subgraph_->Output(bn);
    std::string blob_name = BlobName(lbi);
    CHECK_GT(shapes.count(blob_name), 0);
    BlobDesc *blob_desc = GetBlobDesc4BnInOp(bn);
    blob_desc->set_data_type(shapes.at(blob_name).data_type());
    blob_desc->mut_shape() = Shape(shapes.at(blob_name).shape());
  }
  return Maybe<void>::Ok();
}

Maybe<void> XlaLaunchOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
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
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    XlaLaunchOp::SbpInferHint4IbnFunc SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  const auto &resource_scope = xla_launch_conf.attr().resource_scope();
  const auto &sbp_signatures = resource_scope.sbp_signatures();
  for (const std::string &bn : this->input_bns()) {
    CHECK_GT(sbp_signatures.count(bn), 0);
  }
  for (const std::string &bn : this->output_bns()) {
    CHECK_GT(sbp_signatures.count(bn), 0);
  }

  auto *bn2sbp_parallel = sbp_signature->mutable_bn_in_op2sbp_parallel();
  *bn2sbp_parallel = sbp_signatures;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kXlaLaunchConf, XlaLaunchOp);

}  // namespace oneflow
