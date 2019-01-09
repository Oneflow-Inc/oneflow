#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

std::string OpEdge::VisualStr() const {
  std::string str;
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : lbis_) {
    if (idx++ > 0) { str += ","; }
    str += lbi.blob_name();
  }
  return str;
}

std::string OpNode::VisualStr() const {
  std::string str = op().op_name();
  {
    for (int64_t machine_id : parallel_desc().sorted_machine_ids()) {
      std::string dev_type;
      if (parallel_desc().device_type() == DeviceType::kCPU) {
        dev_type = "cpu";
      } else if (parallel_desc().device_type() == DeviceType::kGPU) {
        dev_type = "gpu";
      } else {
        UNIMPLEMENTED();
      }
      std::string parallel_desc_str = std::to_string(machine_id) + ":" + dev_type + ":";
      const auto& dev_phy_ids = parallel_desc().sorted_dev_phy_ids(machine_id);
      parallel_desc_str += std::to_string(dev_phy_ids.front());
      if (dev_phy_ids.back() > dev_phy_ids.front()) {
        parallel_desc_str += "-" + std::to_string(dev_phy_ids.back());
      }
      str += "\\n" + parallel_desc_str;
    }
  }
  auto GetTimeShapeStr = [&](const Shape& shape, const std::string& prefix) {
    std::string time_shape_str = prefix + ":";
    time_shape_str += "(";
    int32_t idx = 0;
    for (int64_t dim : shape.dim_vec()) {
      if (idx++ > 0) { time_shape_str += ","; }
      time_shape_str += std::to_string(dim);
    }
    time_shape_str += ")";
    return time_shape_str;
  };
  if (in_edges().empty() == false) {
    str += "\\n" + GetTimeShapeStr(*GetInputBlobTimeShape(), "in_blob_time_shape");
  }
  str += "\\n" + GetTimeShapeStr(out_blob_time_shape(), "out_blob_time_shape");
  return str;
}

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

const Shape* OpNode::GetInputBlobTimeShape(const std::string& bn_in_op) const {
  const LogicalBlobId& lbi = op().BnInOp2Lbi(bn_in_op);
  if (ibns_.find(bn_in_op) != ibns_.end()) {
    for (OpEdge* edge : in_edges()) {
      for (const LogicalBlobId& edge_lbi : edge->lbis()) {
        if (lbi == edge_lbi) { return &edge->src_node()->out_blob_time_shape(); }
      }
    }
  }
  UNIMPLEMENTED();
  return nullptr;
}

const Shape* OpNode::GetInputBlobTimeShape() const {
  if (in_edges().empty()) { UNIMPLEMENTED(); }
  OpNode* first_input = (*in_edges().begin())->src_node();
  for (OpEdge* edge : in_edges()) {
    CHECK_EQ(first_input->out_blob_time_shape(), edge->src_node()->out_blob_time_shape());
  }
  return &first_input->out_blob_time_shape();
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
  InferTimeShape();
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
      Connect(producer, NewEdge(lbis, lbi2ibns), op_node);
    }
  });
}

void OpGraph::UpdateOpNodeHasInDiff() {
  TopoForEachNode([&](OpNode* op_node) {
    bool has_diff = false;
    for (OpEdge* edge : op_node->in_edges()) {
      if (edge->src_node()->has_in_diff() || edge->src_node()->has_model_diff()) {
        edge->set_has_diff(true);
        has_diff = true;
        break;
      }
    }
    op_node->set_has_in_diff(has_diff);
  });
}

void OpGraph::InferNodeBlobDesc() const {
  TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    op_node->op().InferBlobDescsIf(
        std::bind(&OpNode::BlobDesc4BnInOp, op_node, std::placeholders::_1), &parallel_ctx,
        [](OpContext*) {});
  });
}

void OpGraph::InferTimeShape() const {
  TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(op_node->parallel_desc().parallel_num());
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    auto GetInputBlobTimeShape = [&](const std::string& bn_in_op) {
      return op_node->GetInputBlobTimeShape(bn_in_op);
    };
    op_node->op().InferOutBlobTimeShape(GetInputBlobTimeShape, &parallel_ctx,
                                        op_node->mut_out_blob_time_shape());
  });
}

void OpGraph::ForEachPseudoChain(
    const std::function<void(const HashSet<OpNode*>&)>& Handler) const {
  auto IsReachable = MakePredicatorIsReachable();
  ForEachComponentWithSameDataParallelDescAndTimeShape(
      [&](const std::vector<OpNode*>& nodes) { ForEachPseudoChain(nodes, IsReachable, Handler); });
}

std::function<bool(OpNode* src, OpNode* dst)> OpGraph::MakePredicatorIsReachable() const {
  auto node2ancestors_ptr = std::make_shared<HashMap<OpNode*, HashSet<OpNode*>>>();
  TopoForEachNode([&](OpNode* node) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      (*node2ancestors_ptr)[node].insert(in_node);
      (*node2ancestors_ptr)[node].insert((*node2ancestors_ptr)[in_node].begin(),
                                         (*node2ancestors_ptr)[in_node].end());
    });
  });
  return [node2ancestors_ptr](OpNode* src, OpNode* dst) -> bool {
    return node2ancestors_ptr->at(dst).find(src) != node2ancestors_ptr->at(dst).end();
  };
}

void OpGraph::ForEachComponentWithSameDataParallelDescAndTimeShape(
    const std::function<void(const std::vector<OpNode*>&)>& Handler) const {
  auto WithSameDataParallelDescAndTimeShape = [](OpNode* src, OpNode* dst) -> bool {
    if (src->parallel_desc().policy() != ParallelPolicy::kDataParallel) { return false; }
    if (dst->parallel_desc().policy() != ParallelPolicy::kDataParallel) { return false; }
    if (src->in_edges().empty()) { return false; }
    if (*src->GetInputBlobTimeShape() != src->out_blob_time_shape()) { return false; }
    if (*dst->GetInputBlobTimeShape() != dst->out_blob_time_shape()) { return false; }
    return src->parallel_desc() == dst->parallel_desc()
           && src->out_blob_time_shape() == dst->out_blob_time_shape();
  };
  auto ForEachNext = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (WithSameDataParallelDescAndTimeShape(in_node, node)) { Handler(in_node); }
    });
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (WithSameDataParallelDescAndTimeShape(node, out_node)) { Handler(out_node); }
    });
  };
  HashMap<OpNode*, int32_t> op_node2component_id;
  int32_t cur_component_id = 0;
  ForEachNode([&](OpNode* start) {
    if (op_node2component_id.find(start) != op_node2component_id.end()) { return; }
    ++cur_component_id;
    BfsForEachNode({start}, ForEachNext, [&](OpNode* node) {
      CHECK(op_node2component_id.emplace(node, cur_component_id).second);
    });
  });
  HashMap<int32_t, std::vector<OpNode*>> component_id2op_nodes;
  for (const auto& pair : op_node2component_id) {
    component_id2op_nodes[pair.second].push_back(pair.first);
  }
  for (const auto& pair : component_id2op_nodes) { Handler(pair.second); }
}

void OpGraph::ForEachPseudoChain(
    const std::vector<OpNode*>& nodes,
    const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable,
    const std::function<void(const HashSet<OpNode*>&)>& Handler) const {
  if (nodes.size() <= 1) { return; }
  if (nodes.front()->parallel_desc().device_type() == DeviceType::kCPU) { return; }
  if (nodes.front()->parallel_desc().policy() != ParallelPolicy::kDataParallel) { return; }
  HashSet<OpNode*> all_nodes(nodes.begin(), nodes.end());
  while (all_nodes.size() > 1) {
    HashSet<OpNode*> chain;
    ReverseTopoGetPseudoChain(all_nodes, &chain, IsReachable);
    Handler(chain);
    for (OpNode* node_in_chain : chain) { all_nodes.erase(node_in_chain); }
  }
}

void OpGraph::ReverseTopoGetPseudoChain(
    const HashSet<OpNode*>& op_nodes, HashSet<OpNode*>* pseudo_chain_nodes,
    const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable) const {
  // get sink nodes
  std::list<OpNode*> sinks;
  auto IsSink = [&](OpNode* node) {
    for (OpNode* inner_node : op_nodes) {
      if (IsReachable(node, inner_node)) { return false; }
    }
    return true;
  };
  for (OpNode* op_node : op_nodes) {
    if (IsSink(op_node)) { sinks.push_back(op_node); }
  }
  // generate connections of subgraph
  HashMap<OpNode*, std::vector<OpNode*>> node2in_nodes;
  HashMap<OpNode*, std::vector<OpNode*>> node2out_nodes;
  auto IsInSubset = [&](OpNode* node) { return op_nodes.find(node) != op_nodes.end(); };
  auto ReachableToAnySink = [&](OpNode* node) {
    for (OpNode* sink : sinks) {
      if (node == sink) { return true; }
      if (IsReachable(node, sink)) { return true; }
    }
    return false;
  };
  auto AnyOutputNodesNotInSubsetAndReachableIntoSink = [&](OpNode* node) {
    for (OpEdge* edge : node->out_edges()) {
      if (!IsInSubset(edge->dst_node()) && ReachableToAnySink(edge->dst_node())) { return true; }
    }
    return false;
  };
  for (OpNode* node : op_nodes) {
    if (AnyOutputNodesNotInSubsetAndReachableIntoSink(node)) { continue; }
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (IsInSubset(out_node)) {
        node2in_nodes[out_node].push_back(node);
        node2out_nodes[node].push_back(out_node);
      }
    });
  }
  // get chain nodes
  auto ForEachInNode = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    for (OpNode* in_node : node2in_nodes[node]) { Handler(in_node); }
  };
  auto ForEachOutNode = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    for (OpNode* out_node : node2out_nodes[node]) { Handler(out_node); }
  };
  TopoForEachNode(sinks, ForEachOutNode, ForEachInNode,
                  [&](OpNode* node) { CHECK(pseudo_chain_nodes->emplace(node).second); });
}

}  // namespace oneflow
