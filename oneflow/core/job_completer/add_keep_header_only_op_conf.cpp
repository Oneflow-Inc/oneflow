#include "oneflow/core/job_completer/add_keep_header_only_op_conf.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void AddKeepHeaderOnlyOp(const OpGraph& op_graph, JobBuilder* job_builder) {
  std::vector<OperatorConf> op_confs;
  op_graph.TopoForEachNode([&](OpNode* node) {
    const PbRpf<std::string>& ibns = node->op().input_bns();
    std::vector<std::string> header_only_ibns;
    for (const std::string& ibn : ibns) {
      if (node->op().InputBlobModifier4Ibn(ibn).use_header_only()) {
        header_only_ibns.push_back(ibn);
      }
    }
    if (header_only_ibns.empty()) { return; }
    CHECK(node->op().op_conf().has_tick_conf() == false);
    auto OpEdge4Lbi = [node](const LogicalBlobId& lbi) -> OpEdge* {
      for (OpEdge* edge : node->in_edges()) {
        for (const LogicalBlobId& edge_lbi : edge->lbis()) {
          if (lbi == edge_lbi) { return edge; }
        }
      }
      UNIMPLEMENTED();
      return nullptr;
    };
    HashMap<OpNode*, std::vector<std::string>> src_node2ibns;
    for (const std::string& ibn : header_only_ibns) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      OpNode* src_node = OpEdge4Lbi(lbi)->src_node();
      if (src_node->parallel_desc() != node->parallel_desc()) {
        LOG(WARNING) << "can not enable KeepHeaderOnly for " << ibn << " of "
                     << node->op().op_name();
        continue;
      }
      if (src_node->SbpParallel4Lbi(lbi) != node->SbpParallel4BnInOp(ibn)) {
        LOG(WARNING) << "can not enable KeepHeaderOnly for " << ibn << " of "
                     << node->op().op_name();
        continue;
      }
      src_node2ibns[src_node].push_back(ibn);
    }
    OperatorConf dst_op_conf = node->op().op_conf();
    PbMessage* dst_op_type_conf =
        MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
    for (const auto& pair : src_node2ibns) {
      OpNode* src_node = pair.first;
      const std::vector<std::string>& cur_ibns = pair.second;
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(cur_ibns.at(0));
      OpEdge* edge = OpEdge4Lbi(lbi);

      OperatorConf op_conf;
      op_conf.set_name(node->op().op_name() + "-" + src_node->op().op_name() + "-keep_header_only");
      KeepHeaderOnlyOpConf* kho_conf = op_conf.mutable_keep_header_only_conf();
      for (const std::string& ibn : cur_ibns) {
        const LogicalBlobId& cur_lbi = node->op().BnInOp2Lbi(ibn);
        OpEdge* cur_edge = OpEdge4Lbi(cur_lbi);
        CHECK(lbi.op_name() == cur_lbi.op_name());
        CHECK(edge == cur_edge);

        *(kho_conf->mutable_in()->Add()) = GenLogicalBlobName(cur_lbi);
        *(kho_conf->mutable_out()->Add()) = cur_lbi.blob_name();

        std::string lbn = op_conf.name() + "/" + cur_lbi.blob_name();
        ReplaceInputLbnInOpCustomizedConf(dst_op_type_conf, ibn, GenLogicalBlobName(cur_lbi), lbn);
      }
      job_builder->AddOps(src_node->parallel_desc().parallel_conf(),
                          std::vector<OperatorConf>{op_conf});
    }
    // make sure an op_conf can only be udpated once
    job_builder->MutOpsOnlyOnce(std::vector<OperatorConf>{dst_op_conf});
  });
}

}  // namespace oneflow
