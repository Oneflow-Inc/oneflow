#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

std::string OpEdge::VisualStr() const {
  std::string str;
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : lbis_) {
    if (idx++ > 0) { str += "\\n"; }
    str += lbi.blob_name() + ":";
    str += src_node()->NoParallelBlobDesc4Lbi(lbi).shape().ToString();
  }
  return str;
}

const BlobDesc& OpNode::NoParallelBlobDesc4Lbi(const LogicalBlobId& lbi) const {
  return lbi2no_parallel_blob_desc_.at(lbi);
}

const BlobDesc& OpNode::LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const {
  return lbi2logical_blob_desc_.at(lbi);
}

const BlobParallelDesc& OpNode::BlobParallelDesc4Lbi(const LogicalBlobId& lbi) const {
  return lbi2blob_parallel_desc_.at(lbi);
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
    time_shape_str += shape.ToString();
    return time_shape_str;
  };
  if (in_edges().empty() == false) {
    str += "\\n" + GetTimeShapeStr(*GetInputBlobTimeShape(), "in_blob_time_shape");
  }
  str += "\\n" + GetTimeShapeStr(out_blob_time_shape(), "out_blob_time_shape");
  return str;
}

BlobDesc* OpNode::MutNoParallelBlobDesc(const LogicalBlobId& lbi) {
  if (lbi2no_parallel_blob_desc_.find(lbi) == lbi2no_parallel_blob_desc_.end()) {
    lbi2no_parallel_blob_desc_.emplace(lbi, BlobDesc());
  }
  return &lbi2no_parallel_blob_desc_.at(lbi);
}

BlobDesc* OpNode::NoParallelBlobDesc4BnInOp(const std::string& bn_in_op) {
  const LogicalBlobId& lbi = op().BnInOp2Lbi(bn_in_op);
  if (ibns_.find(bn_in_op) != ibns_.end()) {
    for (OpEdge* edge : in_edges()) {
      for (const LogicalBlobId& edge_lbi : edge->lbis()) {
        if (lbi == edge_lbi) { return edge->src_node()->MutNoParallelBlobDesc(lbi); }
      }
    }
    UNIMPLEMENTED();
  }
  return MutNoParallelBlobDesc(lbi);
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
      int64_t elem_cnt = op_node->NoParallelBlobDesc4BnInOp(model_bn)->shape().elem_cnt();
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
  FixOpParallelDesc();
  UpdateOpNodeHasInDiff();
  InferTimeShape();
  InferNodeNoParallelBlobDesc();
  HashMap<LogicalBlobId, int32_t> lbi2model_split_axis;
  InferModelSplitAxis(&lbi2model_split_axis);
  InferBlobParallelDesc(lbi2model_split_axis);
  InferLogicalBlobDesc();
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

void OpGraph::FixOpParallelDesc() const {
  ForEachNode([&](OpNode* node) { node->op().FixParallelDesc(node->mut_parallel_desc()); });
  ForEachNode([&](OpNode* node) {
    OpNode* prev_node = node;
    while (prev_node->op().IsElemWiseOp()) { prev_node = prev_node->SoleInEdge()->src_node(); }
    if (prev_node != node && prev_node->parallel_desc().policy() == kModelParallel) {
      *node->mut_parallel_desc() = prev_node->parallel_desc();
    }
  });
}

void OpGraph::UpdateOpNodeHasInDiff() const {
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

void OpGraph::InferNodeNoParallelBlobDesc() const {
  TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    op_node->op().InferBlobDescsIf(
        std::bind(&OpNode::NoParallelBlobDesc4BnInOp, op_node, std::placeholders::_1),
        &parallel_ctx, job_desc_->RecordPieceSize(), [](OpContext*) {});
  });
}

void OpGraph::InferModelSplitAxis(HashMap<LogicalBlobId, int32_t>* lbi2model_split_axis) const {
  TopoForEachNode([&](OpNode* op_node) {
    auto ModelSplitAxis4BnInOp = [&](const std::string& bn) -> int32_t* {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      if (lbi2model_split_axis->find(lbi) == lbi2model_split_axis->end()) {
        lbi2model_split_axis->emplace(lbi, -1);
      }
      return &lbi2model_split_axis->at(lbi);
    };
    auto ShapeNumAxes4BnInOp = [&](const std::string& bn) -> int32_t {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      return op_node->NoParallelBlobDesc4Lbi(lbi).shape().NumAxes();
    };
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(op_node->parallel_desc().parallel_num());
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    op_node->op().InferBlobModelSplitAxisIf(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                            &parallel_ctx);
  });
}

void OpGraph::InferBlobParallelDesc(
    const HashMap<LogicalBlobId, int32_t>& lbi2model_split_axis) const {
  TopoForEachNode([&](OpNode* op_node) {
    auto BlobParallelDesc4BnInOp = [&](const std::string& bn) -> BlobParallelDesc* {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      return op_node->MutBlobParallelDesc4BnInOp(bn, lbi2model_split_axis.at(lbi));
    };
    auto ProducerBlobParallelDesc4BnInOp = [&](const std::string& bn) -> const BlobParallelDesc& {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      for (OpEdge* in_edge : op_node->in_edges()) {
        if (in_edge->lbi2ibns().find(lbi) != in_edge->lbi2ibns().end()) {
          return in_edge->src_node()->BlobParallelDesc4Lbi(lbi);
        }
      }
    };
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(op_node->parallel_desc().parallel_num());
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    op_node->op().InferBlobParallelDescIf(BlobParallelDesc4BnInOp, ProducerBlobParallelDesc4BnInOp,
                                          &parallel_ctx);
  });
}

void OpGraph::InferLogicalBlobDesc() const { TODO(); }

int32_t OpGraph::GetModelSplitAxis(const std::string& op_name, const LogicalBlobId& lbi) const {
  const auto& blob_parallel_desc = GetBlobParallelDesc(op_name, lbi);
  CHECK(blob_parallel_desc.has_model_split_axis());
  return blob_parallel_desc.model_split_axis();
}

int64_t OpGraph::GetModelSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const {
  OpNode* op_node = op_name2op_node_.at(GetOpNameKey(op_name, lbi));
  const LogicalBlobId& lbi_key = GetLogicalBlobIdKey(op_name, lbi);
  const BlobParallelDesc& blob_parallel_desc = op_node->BlobParallelDesc4Lbi(lbi_key);
  CHECK(blob_parallel_desc.has_model_split_axis());
  return op_node->LogicalBlobDesc4Lbi(lbi_key).shape().At(blob_parallel_desc.model_split_axis());
}
int64_t OpGraph::GetDataSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const {
  OpNode* op_node = op_name2op_node_.at(GetOpNameKey(op_name, lbi));
  const LogicalBlobId& lbi_key = GetLogicalBlobIdKey(op_name, lbi);
  const BlobParallelDesc& blob_parallel_desc = op_node->BlobParallelDesc4Lbi(lbi_key);
  CHECK(blob_parallel_desc.has_data_parallel());
  return op_node->LogicalBlobDesc4Lbi(lbi_key).shape().At(0);
}
int64_t OpGraph::GetParallelNum(const std::string& op_name, const LogicalBlobId& lbi) const {
  return GetBlobParallelDesc(op_name, lbi).ParallelNum();
}
const BlobParallelDesc& OpGraph::GetBlobParallelDesc(const std::string& op_name,
                                                     const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(GetOpNameKey(op_name, lbi))
      ->BlobParallelDesc4Lbi(GetLogicalBlobIdKey(op_name, lbi));
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

std::string OpGraph::GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const {
  CHECK(!lbi.has_is_packed_id());
  std::string op_name_key;
  if (op_name2op_node_.find(op_name) == op_name2op_node_.end()) {
    CHECK(lbi.has_clone_id());
    return lbi.op_name();
  } else {
    CHECK(!lbi.has_clone_id());
    return op_name;
  }
}

LogicalBlobId OpGraph::GetLogicalBlobIdKey(const std::string& op_name,
                                           const LogicalBlobId& lbi) const {
  CHECK(!lbi.has_is_packed_id());
  if (op_name2op_node_.find(op_name) == op_name2op_node_.end()) {
    CHECK(lbi.has_clone_id());
    LogicalBlobId lbi_key;
    lbi_key.set_op_name(lbi.op_name());
    lbi_key.set_blob_name(lbi.blob_name());
    return lbi_key;
  } else {
    CHECK(!lbi.has_clone_id());
    return lbi;
  }
}

}  // namespace oneflow
