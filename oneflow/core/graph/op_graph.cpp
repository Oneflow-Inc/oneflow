#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

BlobDesc* OpNode::MutBlobDesc(const LogicalBlobId& lbi) {
  if (lbi2blob_desc_.find(lbi) == lbi2blob_desc_.end()) {
    lbi2blob_desc_.emplace(lbi, std::make_shared<BlobDesc>());
  }
  return lbi2blob_desc_.at(lbi).get();
}

BlobDesc* OpNode::BlobDesc4BnInOp(const std::string& bn_in_op) {
  const LogicalBlobId& lbi = op().BnInOp2Lbi(bn_in_op);
  if (ibns_.find(bn_in_op) != ibns_.end()) {
    for (OpEdge* edge : in_edges()) {
      for (const LogicalBlobId& edge_lbi : edge->lbis()) {
        if (lbi == edge_lbi) { return edge->src_node()->MutBlobDesc(lbi); }
      }
    }
    UNIMPLEMENTED();
  }
  return MutBlobDesc(lbi);
}

void OpGraph::InferOpModelSize(HashMap<std::string, size_t>* op_name2model_size) {
  ForEachNode([&](OpNode* op_node) {
    size_t model_size = 0;
    for (const std::string& model_bn : op_node->op().model_bns()) {
      int64_t elem_cnt = op_node->BlobDesc4BnInOp(model_bn)->shape().elem_cnt();
      model_size += elem_cnt * GetSizeOfDataType(job_desc_->DefaultDataType());
      model_size = RoundUp(model_size, kCudaAlignSize);
    }
    size_t parallel_num = op_node->parallel_desc().parallel_num();
    if (op_node->parallel_desc().policy() == ParallelPolicy::kModelParallel) {
      model_size = (model_size + parallel_num - 1) / parallel_num;
    }
    CHECK(op_name2model_size->emplace(op_node->op().op_name(), model_size).second);
  });
}

void OpGraph::Init() {
  InitNodes();
  ForEachNode(
      [&](OpNode* node) { CHECK(op_name2op_node_.emplace(node->op().op_name(), node).second); });
  InitEdges();
  UpdateOpNodeHasInDiff();
  InferNodeBlobDesc();
}

void OpGraph::InitNodes() {
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_desc_->placement());
  for (const auto& op_conf : job_desc_->dlnet_conf().op()) {
    OpNode* node = new OpNode(ParallelDesc(*ParallelConf4OpName(op_conf.name())), op_conf);
    AddAllocatedNode(node);
  }
}

void OpGraph::InitEdges() {
  HashMap<LogicalBlobId, OpNode*> lbi2producer;
  ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      CHECK(lbi2producer.emplace(op_node->op().BnInOp2Lbi(obn), op_node).second);
    }
  });
  ForEachNode([&](OpNode* op_node) {
    HashMap<std::string, std::vector<LogicalBlobId>> producer_name2lbis;
    HashMap<std::string, HashMap<LogicalBlobId, std::vector<std::string>>>
        consumer_op_name2lbi2ibns;
    for (const auto& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      producer_name2lbis[lbi.op_name()].push_back(lbi);
      consumer_op_name2lbi2ibns[op_node->op().op_name()][lbi].push_back(ibn);
    }
    for (const auto& pair : producer_name2lbis) {
      const auto& lbis = pair.second;
      const auto& lbi2ibns = consumer_op_name2lbi2ibns.at(op_node->op().op_name());
      OpNode* producer = lbi2producer.at(lbis.at(0));
      Connect(producer, new OpEdge(lbis, lbi2ibns), op_node);
    }
  });
}

void OpGraph::UpdateOpNodeHasInDiff() {
  auto HasIndiff = [&](const OpNode* op_node) -> bool {
    for (OpEdge* edge : op_node->in_edges()) {
      if (edge->src_node()->has_in_diff()) { return true; }
      if (edge->src_node()->has_model_diff()) { return true; }
    }
    return false;
  };
  TopoForEachNode([&](OpNode* op_node) { op_node->set_has_in_diff(HasIndiff(op_node)); });
}

void OpGraph::InferNodeBlobDesc() const {
  TopoForEachNode([&](OpNode* op_node) {
    auto GetBlobDesc4BnInOp = [&](const std::string& bn_in_op) -> BlobDesc* {
      return op_node->BlobDesc4BnInOp(bn_in_op);
    };
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    op_node->op().InferBlobDescsIf(GetBlobDesc4BnInOp, &parallel_ctx, [](OpContext*) {});
  });
}

}  // namespace oneflow
