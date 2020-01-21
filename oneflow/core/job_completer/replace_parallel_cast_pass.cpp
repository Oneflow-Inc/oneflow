#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

class ReplaceParallelCastPass final : public OpGraphPass {
 public:
  ReplaceParallelCastPass() = default;
  ~ReplaceParallelCastPass() = default;
  bool IsEnabled() const override { return true; }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

void ReplaceParallelCastPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_parallel_cast_conf()) { return; }
    const ParallelCastOpConf& parallel_cast_conf = op_conf.parallel_cast_conf();
    if (!parallel_cast_conf.has_split_axis()) { return; }
    if (op_node->out_edges().size() != 1) { return; }
    const OpNode* dst_node = op_node->SoleOutEdge()->dst_node();
    if (dst_node->in_edges().size() != 1) { return; }
    OperatorConf dst_op_conf = dst_node->op().op_conf();
    const LogicalBlobId& parallel_cast_in_lbi = op_node->op().BnInOp2Lbi("in");
    const LogicalBlobId& parallel_cast_out_lbi = op_node->op().BnInOp2Lbi("out");
    const SbpParallel& sbp_parallel = op_node->SbpParallel4Lbi(parallel_cast_in_lbi);
    if (dst_node->SbpParallel4Lbi(parallel_cast_out_lbi) != sbp_parallel) { return; }
    PbMessage* conf = MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
    ReplaceStrValInPbFdOrPbRpf(conf, dst_node->op().SoleIbn(),
                               GenLogicalBlobName(parallel_cast_out_lbi),
                               GenLogicalBlobName(parallel_cast_in_lbi));
    job_builder->MutOpsOnlyOnce({dst_op_conf});
    job_builder->DelOps({op_conf});
    SbpSignature sbp_signature{};
    (*sbp_signature.mutable_bn_in_op2sbp_parallel())[dst_node->op().SoleIbn()] = sbp_parallel;
    job_builder->AddSbpSignature4OpName(dst_node->op().op_name(), sbp_signature);
  });
}

}  // namespace

REGISTER_FUNCTION_PASS("ReplaceParallelCastPass", ReplaceParallelCastPass);

}  // namespace oneflow
