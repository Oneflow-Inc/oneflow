#include "oneflow/core/job_completer/auto_mixed_precision.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

template<typename MapT, typename KeyT>
bool IsKeyFound(const MapT& m, const KeyT& k) {
  return m.find(k) != m.end();
}

template<typename GraphType, typename NodeType>
void DfsGraphTraversal(
    const GraphType& graph, const std::list<NodeType*>& starts,
    std::function<void(NodeType*, std::function<void(NodeType*)>)> ForEachNextNode,
    std::function<bool(NodeType*)> IsNodeTraversable, std::function<void(NodeType*)> Handler) {
  HashSet<NodeType*> visited;
  std::stack<NodeType*> stack;
  for (NodeType* start : starts) { stack.push(start); }
  while (!stack.empty()) {
    NodeType* cur_node = stack.top();
    stack.pop();

    if (IsNodeTraversable(cur_node) && !IsKeyFound(visited, cur_node)) {
      Handler(cur_node);
      visited.insert(cur_node);
      ForEachNextNode(cur_node, [&stack](NodeType* next) { stack.push(next); });
    }
  }
}

std::function<bool(OpNode*)> MakePredicatorIsAllowedToRunWithHalf(const OpGraph& op_graph) {
  std::shared_ptr<HashSet<OpNode*>> allowed_set;
  op_graph.ForEachNode([&](OpNode* node) {
    if (node->parallel_desc().device_type() != DeviceType::kGPU) { return; }
    for (const std::string& obn : node->op().output_bns()) {
      LogicalBlobId lbi = node->op().BnInOp2Lbi(obn);
      // TODO(niuchong): this ain't right for fw-bw-opgraph, but right for fw-opgraph
      if (node->HasBatchDim4Lbi(lbi) == false) { return; }
    }
    allowed_set->insert(node);
  });
  return [allowed_set](OpNode* node) -> bool { return IsKeyFound(*allowed_set, node); };
}

bool IsOpFloat32(const OpNode* node) {
  for (const std::string& obn : node->op().output_bns()) {
    LogicalBlobId lbi = node->op().BnInOp2Lbi(obn);
    const BlobDesc& blob_desc = node->LogicalBlobDesc4Lbi(lbi);
    if (blob_desc.data_type() != DataType::kFloat) { return false; }
  }
  return true;
}

}  // namespace

void AutoMixedPrecision::Apply(const OpGraph& op_graph, Job* job) const {
  CHECK(Global<JobDesc>::Get()->DefaultDataType() == DataType::kFloat);
  HashSet<OpNode*> black_set;
  HashSet<OpNode*> white_set;
  SetBlackSet(op_graph, &black_set);

  auto IsAllowedToRunWithHalf = MakePredicatorIsAllowedToRunWithHalf(op_graph);
  SetWhiteSet(op_graph, IsAllowedToRunWithHalf, black_set, &white_set);
  InsertCastOp(op_graph, white_set, job);
}

void AutoMixedPrecision::SetBlackSet(const OpGraph& op_graph, HashSet<OpNode*>* black_set) const {
  auto IsNonListNode = [&](OpNode* node) -> bool {
    OperatorConf::OpTypeCase op_type = node->op().op_conf().op_type_case();
    return !IsKeyFound(white_list_, op_type) && !IsKeyFound(black_list_, op_type)
           && !IsKeyFound(gray_list_, op_type);
  };

  HashSet<OpNode*> upstream_or_part_of_black_and_gray;
  op_graph.ForEachNode([&](OpNode* start_node) {
    OperatorConf::OpTypeCase op_type = start_node->op().op_conf().op_type_case();
    if (IsKeyFound(black_list_, op_type) || IsKeyFound(gray_list_, op_type)) {
      DfsGraphTraversal<OpGraph, OpNode>(
          op_graph, std::list<OpNode*>{start_node}, &OpNode::ForEachNodeOnOutEdge,
          [&](OpNode* node) -> bool {
            return (node == start_node)
                   || (!IsKeyFound(upstream_or_part_of_black_and_gray, node)
                       && IsNonListNode(node));
          },
          [&](OpNode* node) {
            upstream_or_part_of_black_and_gray.insert(node);
            LOG(INFO) << "insert " << node->op().op_name()
                      << " to upstream_or_part_of_black_and_gray";
          });
    }
  });

  // propagate black through upstream_or_part_of_black_and_gray
  op_graph.ForEachNode([&](OpNode* start_node) {
    OperatorConf::OpTypeCase op_type = start_node->op().op_conf().op_type_case();
    if (IsKeyFound(*black_set, start_node) || !IsKeyFound(black_list_, op_type)) { return; }
    DfsGraphTraversal<OpGraph, OpNode>(
        op_graph, std::list<OpNode*>{start_node}, &OpNode::ForEachNodeOnInEdge,
        [&](OpNode* node) -> bool {
          return (node == start_node)
                 || (!IsKeyFound(*black_set, node)
                     && IsKeyFound(upstream_or_part_of_black_and_gray, node));
        },
        [&](OpNode* node) {
          black_set->insert(node);
          LOG(INFO) << "insert " << node->op().op_name() << " to black_set";
        });
  });
}

void AutoMixedPrecision::SetWhiteSet(const OpGraph& op_graph,
                                     std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                                     const HashSet<OpNode*>& black_set,
                                     HashSet<OpNode*>* white_set) const {
  HashSet<OpNode*> upstream_or_part_of_white;
  op_graph.ForEachNode([&](OpNode* start_node) {
    OperatorConf::OpTypeCase op_type = start_node->op().op_conf().op_type_case();
    if (IsAllowedToRunWithHalf(start_node) && IsKeyFound(white_list_, op_type)) {
      DfsGraphTraversal<OpGraph, OpNode>(
          op_graph, std::list<OpNode*>{start_node}, &OpNode::ForEachNodeOnOutEdge,
          [&](OpNode* node) -> bool {
            return (node == start_node)
                   || (!IsKeyFound(upstream_or_part_of_white, node) && !IsKeyFound(black_set, node)
                       && IsAllowedToRunWithHalf(node) && IsOpFloat32(node));
          },
          [&](OpNode* node) {
            upstream_or_part_of_white.insert(node);
            LOG(INFO) << "insert " << node->op().op_name() << " to upstream_or_part_of_white";
          });
    }
  });

  op_graph.ForEachNode([&](OpNode* start_node) {
    OperatorConf::OpTypeCase op_type = start_node->op().op_conf().op_type_case();
    if (IsAllowedToRunWithHalf(start_node) && !IsKeyFound(*white_set, start_node)
        && IsKeyFound(white_list_, op_type)) {
      DfsGraphTraversal<OpGraph, OpNode>(
          op_graph, std::list<OpNode*>{start_node}, &OpNode::ForEachNodeOnInEdge,
          [&](OpNode* node) -> bool {
            return (node == start_node)
                   || (!IsKeyFound(*white_set, node)
                       && IsKeyFound(upstream_or_part_of_white, node));
          },
          [&](OpNode* node) {
            white_set->insert(node);
            LOG(INFO) << "insert " << node->op().op_name() << " to white_set";
          });
    }
  });
}

void AutoMixedPrecision::InsertCastOp(const OpGraph& op_graph, const HashSet<OpNode*>& white_set,
                                      Job* job) const {
  JobBuilder job_builder(job);

  HashSet<OpEdge*> white_set_in_edges;
  op_graph.ForEachNode([&](OpNode* node) {
    if (IsKeyFound(white_set, node)) {
      for (OpEdge* edge : node->in_edges()) {
        if (!IsKeyFound(white_set, edge->src_node())) { white_set_in_edges.insert(edge); }
      }
    }
  });
  HashMap<std::string, OperatorConf> dst_op_name2dst_op_confs;
  for (OpEdge* edge : white_set_in_edges) {
    CHECK_EQ(1, edge->lbis().size());
    LogicalBlobId cur_lbi = edge->lbis().front();
    CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
    const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();
    OpNode* src_node = edge->src_node();
    OpNode* dst_node = edge->dst_node();

    OperatorConf cast_op_conf;
    cast_op_conf.set_name(src_node->op().op_name() + "-" + dst_node->op().op_name() + "-cast_f2h");
    CastOpConf* cast_conf = cast_op_conf.mutable_cast_conf();
    cast_conf->set_in(GenLogicalBlobName(cur_lbi));
    cast_conf->set_out("out");
    cast_conf->set_data_type(DataType::kFloat16);
    job_builder.AddOps(src_node->parallel_desc().parallel_conf(),
                       std::vector<OperatorConf>{cast_op_conf});

    const std::string& dst_op_name = dst_node->op().op_name();
    if (!IsKeyFound(dst_op_name2dst_op_confs, dst_op_name)) {
      dst_op_name2dst_op_confs.insert(std::make_pair(dst_op_name, dst_node->op().op_conf()));
    }
    OperatorConf dst_op_conf = dst_op_name2dst_op_confs.at(dst_op_name);
    PbMessage* dst_op_type_conf =
        MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
    std::string lbn = cast_op_conf.name() + "/out";
    SetBnValInOpTypeConf(dst_op_type_conf, dst_ibn, GenLogicalBlobName(cur_lbi), lbn);
  }
  std::vector<OperatorConf> dst_op_confs;
  for (const auto& pair : dst_op_name2dst_op_confs) { dst_op_confs.push_back(pair.second); }
  // make sure an op_conf can only be udpated once, cuz later update will override before
  job_builder.MutOps(dst_op_confs);
}

}  // namespace oneflow
