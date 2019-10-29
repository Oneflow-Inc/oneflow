#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/xrt/launch_op.h"
#include "oneflow/xrt/launch_util.h"
#include "oneflow/xrt/xrt_api.h"

namespace oneflow {

void XrtLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xrt_launch_conf());
  const auto &launch_conf = op_conf().xrt_launch_conf();
  int inputs_num = launch_conf.in().size();
  int outputs_num = launch_conf.out().size();

  for (int i = 0; i < inputs_num; ++i) {
    const std::string &input = launch_conf.in().at(i);
    bool mutability = xrt::LookupMutability(launch_conf, input);
    EnrollInputBn(absl::StrCat("in_", i))->set_is_mutable(mutability);
  }
  if (outputs_num > 0) {
    EnrollRepeatedOutputBn("out");
  }
}

const PbMessage &XrtLaunchOp::GetCustomizedConf() const {
  return op_conf().xrt_launch_conf();
}

Maybe<void> XrtLaunchOp::InferBlobDescs(
    std::function<BlobDesc *(const std::string &)> GetBlobDesc4BnInOp,
    const ParallelContext *parallel_ctx) const {
  // Prepare outer input blob descs
  std::unordered_map<std::string, BlobDesc> blob_descs;
  for (const std::string &bn : this->input_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);
    blob_descs[blob_name].CopyMetaFrom(*GetBlobDesc4BnInOp(bn));
  }

  // Build graph from launch conf, and inference output shape
  {
    const auto &launch_conf = op_conf().xrt_launch_conf();
    // Run InferShape pass
    auto options = xrt::CreateDefaultXrtPassOptions();
    auto graph = xrt::BuildXrtGraph(launch_conf, op_conf().device_type(),
                                    this->job_desc());
    xrt::RunXrtPass("InferShape", graph.get(), options, &this->job_desc(),
                    &blob_descs);
  }

  // Fetch output blob descs
  for (const std::string &bn : this->output_bns()) {
    CHECK_GT(blob_descs.count(bn), 0);
    *GetBlobDesc4BnInOp(bn) = blob_descs[bn];
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOp::InferBatchAxis(
    std::function<OptInt64 *(const std::string &)> BatchAxis4BnInOp) const {
  const auto &launch_conf = op_conf().xrt_launch_conf();
  const auto &batch_axis = launch_conf.attr().batch_axis();

  xrt::LaunchGraphHelper graph(launch_conf.attr());
  for (const std::string &bn : this->output_bns()) {
    std::string blob_name = graph.Output(bn);
    CHECK_GT(batch_axis.count(blob_name), 0);
    *BatchAxis4BnInOp(bn) = batch_axis.at(blob_name);
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOp::InferSbpSignature(
    SbpSignature *sbp_signature, const SbpSignature &sbp_sig_conf,
    const std::function<int32_t(const SbpSignature &)> &CalcOrderValue4SbpSig,
    XrtLaunchOp::SbpInferHint4IbnFunc SbpInferHint4Ibn,
    const ParallelDesc &parallel_desc) const {
  const auto &bn2sbp_parallel = sbp_sig_conf.bn_in_op2sbp_parallel();
  for (const std::string &bn : this->input_bns()) {
    CHECK_GT(bn2sbp_parallel.count(bn), 0);
  }
  for (const std::string &bn : this->output_bns()) {
    CHECK_GT(bn2sbp_parallel.count(bn), 0);
  }
  *sbp_signature = sbp_sig_conf;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kXrtLaunchConf, XrtLaunchOp);

}  // namespace oneflow
