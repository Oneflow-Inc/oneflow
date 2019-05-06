#include "oneflow/core/job_completer/freeze_sbp_signature.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

void FreezeSbpSignature(const OpGraph& op_graph, Job* job) {
  op_graph.ForEachNode([&](const OpNode* node) {
    (*job->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf())[node->op().op_name()] =
        node->sbp_signature();
  });
}

}  // namespace oneflow
