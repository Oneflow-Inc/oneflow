#include "oneflow/core/graph/op_graph.h"
#include "op_graph.h"

namespace oneflow {

std::string OpNode::VisualStr() const {
  std::ostringstream oss;
  oss << "OpNode";
  for (const std::shared_ptr<Operator>& op : op_vec_) { oss << "\\n" << op->op_name(); }
  return oss.str();
}

OpGraph::OpGraph(const DLNetConf& net_conf, const Placement& placement) {
  BuildFwStruct(net_conf, placement);
  ToDotWithAutoFilePath();
}

OpGraph::OpGraph()
    : OpGraph(Global<JobDesc>::Get()->dlnet_conf(), Global<JobDesc>::Get()->placement()) {}

void OpGraph::BuildFwStruct(const DLNetConf& net_conf, const Placement& placement) {
  HashMap<std::string, std::shared_ptr<ParallelDesc>> op_name2parallel_desc;
  for (const PlacementGroup& group : placement.placement_group()) {
    for (const std::string& op_name : group.op_set().op_name()) {
      CHECK(op_name2parallel_desc
                .emplace(op_name, std::make_shared<ParallelDesc>(group.parallel_conf()))
                .second);
    }
  }

  HashMap<LogicalBlobId, OpNode*> lbi2producer;
  for (OperatorConf op_conf : net_conf.op()) {
    auto parallel_desc_ptr_it = op_name2parallel_desc.find(op_conf.name());
    CHECK(parallel_desc_ptr_it != op_name2parallel_desc.end());
    const std::shared_ptr<ParallelDesc> parallel_desc = parallel_desc_ptr_it->second;

    op_conf.set_device_type(parallel_desc->device_type());
    std::shared_ptr<Operator> op = ConstructOp(op_conf);
    op->FixParallelDesc(parallel_desc.get());

    OpNode* op_node = NewNode();
    op_node->mut_op_vec() = {op};
    op_node->mut_parallel_desc() = parallel_desc;

    for (const std::string& obn : op->output_bns()) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, op_node).second);
    }
  }

  HashMap<std::pair<OpNode*, OpNode*>, std::vector<LogicalBlobId>> pred_succ2lbis;
  ForEachNode([&](OpNode* cur_node) {
    for (const auto& op : cur_node->op_vec()) {
      for (const std::string& ibn : op->input_bns()) {
        const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
        OpNode* pred_node = lbi2producer.at(lbi);
        if (pred_node == cur_node) { continue; }
        pred_succ2lbis[std::make_pair(pred_node, cur_node)].emplace_back(lbi);
      }
    }
  });

  for (auto& pair : pred_succ2lbis) {
    OpEdge* edge = NewEdge();
    SortAndRemoveDuplication(&pair.second);
    edge->mut_lbi_vec() = pair.second;
    Connect(pair.first.first, edge, pair.first.second);
  }
}

}  // namespace oneflow
