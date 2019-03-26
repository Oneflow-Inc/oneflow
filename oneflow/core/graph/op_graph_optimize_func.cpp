#include "oneflow/core/graph/op_graph_optimize_func.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void AddKeepHeaderOnlyOp(const OpGraph& op_graph, JobConf1* job_conf) {
  JobConfBuilder job_conf_builder(job_conf);

  std::vector<OperatorConf> op_confs;
  op_graph.TopoForEachNode([&](OpNode* node) {
    std::vector<std::string> ibns = node->op().GetHeaderOnlyIbns();
    if (ibns.empty()) { return; }

    auto OpEdge4Lbi = [node](const LogicalBlobId& lbi) -> OpEdge* {
      for (OpEdge* edge : node->in_edges()) {
        for (const LogicalBlobId& edge_lbi : edge->lbis()) {
          if (lbi == edge_lbi) { return edge; }
        }
      }
    };
    for (const std::string& ibn : ibns) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      OpEdge* edge = OpEdge4Lbi(lbi);

      OperatorConf kho_conf;
      kho_conf.set_name(node->op().op_name() + "-" + ibn + "-keep_header_only");
      kho_conf.mutable_keep_header_only_conf()->set_in(GenLogicalBlobName(lbi));
      kho_conf.mutable_keep_header_only_conf()->set_out("out");
      job_conf_builder.AddOps(
          edge->src_node()->parallel_desc().parallel_conf(), std::vector<OperatorConf>{kho_conf});

      OperatorConf dst_op_conf = edge->dst_node()->op().op_conf();
      PbMessage* op_type_conf = MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
      std::string lbn = kho_conf.name() + "/out";
      SetBnValInOpTypeConf(op_type_conf, ibn, kho_conf.keep_header_only_conf().in(), lbn);
      job_conf_builder.MutOps(std::vector<OperatorConf>{dst_op_conf});
    }
  });
}

} // namespace oneflow
