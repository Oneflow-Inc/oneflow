#include "absl/strings/str_split.h"
#include "oneflow/core/compiler/of2xla/xla_launch_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace mola {
extern const std::string _XlaInArgumentPrefix;
extern const std::string _XlaOutArgumentPrefix;
}  // namespace mola

void XlaLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xla_launch_conf());

  EnrollRepeatedInputBn("in");
  EnrollRepeatedOutputBn("out");
  // Mapping the outer inputs and outputs with that of the subgraph arguments
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  for (const auto &arg_conf : xla_launch_conf.attr().argument()) {
    const auto &inputs = arg_conf.in();
    const auto &outputs = arg_conf.out();
    CHECK_EQ(inputs.size(), outputs.size());
    if (absl::StartsWith(arg_conf.name(), mola::_XlaInArgumentPrefix)) {
      for (int i = 0; i < inputs.size(); ++i) {
        subgraph_inputs_.emplace(outputs[i], inputs[i]);
      }
    }
    if (absl::StartsWith(arg_conf.name(), mola::_XlaOutArgumentPrefix)) {
      for (int i = 0; i < inputs.size(); ++i) {
        subgraph_outputs_.emplace(outputs[i], inputs[i]);
      }
    }
  }
}

const PbMessage &XlaLaunchOp::GetCustomizedConf() const {
  return op_conf().xla_launch_conf();
}

void XlaLaunchOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  // Prepare outer input blob descs
  std::unordered_map<std::string, BlobDesc> blob_descs;
  for (const std::string &bn : this->input_bns()) {
    const LogicalBlobId& lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = GenLogicalBlobName(lbi);
    const auto &it = subgraph_inputs_.find(blob_name);
    CHECK(it != subgraph_inputs_.end());
    blob_descs.emplace(it->second, *GetBlobDesc4BnInOp(bn));
  }
  // Inference blob descs in subgraph
  // TODO(hjchen2) Need to topology traversal the attribute's nodes
  for (const auto &node_conf : xla_launch_conf.attr().node()) {
    std::shared_ptr<Operator> node_op = ConstructOp(node_conf);
    auto SubGraphGetBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
      const LogicalBlobId &lbi = node_op->BnInOp2Lbi(bn);
      std::string blob_name = GenLogicalBlobName(lbi);
      const auto &input_bns = node_op->input_bns();
      auto it = blob_descs.find(blob_name);
      // Check presentness for inputs, or create for outputs
      if (std::find(input_bns.begin(), input_bns.end(), bn) !=
          input_bns.end()) {
        CHECK(it != blob_descs.end());
      } else {
        if (it == blob_descs.end()) {
          it = blob_descs.emplace(blob_name, BlobDesc()).first;
        }
      }
      return &(it->second);
    };
    node_op->InferBlobDescs(SubGraphGetBlobDesc4BnInOp, parallel_ctx);
  }
  // Fetch output blob descs
  for (const std::string &bn : this->output_bns()) {
    const auto &bn_it = subgraph_outputs_.find(bn);
    CHECK(bn_it != subgraph_outputs_.end());
    const auto &blob_it = blob_descs.find(bn_it->second);
    CHECK(blob_it != blob_descs.end());
    *GetBlobDesc4BnInOp(bn) = blob_it->second;
  }
}

void XlaLaunchOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  const auto &batch_dim_blobs = xla_launch_conf.attr().batch_dim_blob();
  auto has_batch_dim_fn = [&](const std::string &blob_name) {
    return (std::find(batch_dim_blobs.begin(), batch_dim_blobs.end(), blob_name)
                != batch_dim_blobs.end());
  };
  for (const std::string &bn : this->output_bns()) {
    const auto &it = subgraph_outputs_.find(bn);
    CHECK(it != subgraph_outputs_.end());
    *HasBatchDim4BnInOp(bn) = has_batch_dim_fn(it->second);
  }
}

void XlaLaunchOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  UNIMPLEMENTED();
  // const auto &xla_launch_conf = op_conf().xla_launch_conf();
  // const auto &attr_sbp_conf = xla_launch_conf.attr().sbp_signature();
  // SbpSignature sbp_conf;
  // auto *bn2sbp_parallel = sbp_conf.mutable_bn_in_op2sbp_parallel();
  // for (const std::string &bn : this->input_bns()) {
  //   std::string blob_name = GenLogicalBlobName(this->BnInOp2Lbi(bn));
  //   const auto &it = subgraph_inputs_.find(blob_name);
  //   CHECK(it != subgraph_inputs_.end());
  //   CHECK_GT(attr_sbp_conf.count(it->second), 0);
  //   (*bn2sbp_parallel)[bn] = attr_sbp_conf.at(it->second);
  // }
  // for (const std::string &bn : this->input_bns()) {
  //   const auto &it = subgraph_outputs_.find(bn);
  //   CHECK(it != subgraph_outputs_.end());
  //   CHECK_GT(attr_sbp_conf.count(it->second), 0);
  //   (*bn2sbp_parallel)[bn] = attr_sbp_conf.at(it->second);
  // }
  // this->InferSbpSignature(sbp_signature, sbp_conf, CalcOrderValue4SbpSig,
  //                         SbpInferHint4Ibn, parallel_desc);
}

REGISTER_OP(OperatorConf::kXlaLaunchConf, XlaLaunchOp);

}  // namespace oneflow
