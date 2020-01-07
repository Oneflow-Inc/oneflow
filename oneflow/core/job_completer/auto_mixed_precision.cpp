#include <algorithm>
#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"
#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

#define INSERT_CHECK(expr) CHECK(expr.second)

template<typename MapT, typename KeyT>
bool IsKeyFound(const MapT& m, const KeyT& k) {
  return m.find(k) != m.end();
}

bool IsNodeInList(const AMPList& amp_list, OpNode* node) {
  OperatorConf::OpTypeCase op_type = node->op().op_conf().op_type_case();
  return IsKeyFound(amp_list, op_type);
}

template<typename ContainerT, typename ElemT>
std::string Container2Str(const ContainerT& container,
                          std::function<std::string(const ElemT&)> elem2str) {
  std::string ret;
  bool is_first = true;
  for (const ElemT& elem : container) {
    if (is_first) {
      is_first = false;
    } else {
      ret += ", ";
    }
    ret += elem2str(elem);
  }
  return ret;
}
void DfsTopoGraphTraversal(const OpGraph& graph, bool reversed,
                           std::function<bool(OpNode*)> IsCurNodeStartNode,
                           std::function<bool(OpNode*)> IsCurNodeSatisfied,
                           std::function<bool(OpNode*)> IsFatherNodeSatisfied,
                           std::function<void(OpNode*)> NodeHandler) {
  auto start_nodes = reversed ? graph.sink_nodes() : graph.source_nodes();
  std::function<void(OpNode*, std::function<void(OpNode*)>)> NodeOnInEdge =
      reversed ? &OpNode::ForEachNodeOnOutEdge : &OpNode::ForEachNodeOnInEdge;
  std::function<void(OpNode*, std::function<void(OpNode*)>)> NodeOnOutEdge =
      reversed ? &OpNode::ForEachNodeOnInEdge : &OpNode::ForEachNodeOnOutEdge;
  graph.DfsTopoForEachNode(start_nodes, NodeOnInEdge, NodeOnOutEdge, [&](OpNode* node) {
    if (IsCurNodeStartNode(node)) {
      NodeHandler(node);
      return;
    }
    if (IsCurNodeSatisfied(node)) {
      bool is_one_father_of_node_satisfied = false;
      NodeOnInEdge(node, [&](OpNode* father_node) {
        if (is_one_father_of_node_satisfied) { return; }
        if (IsFatherNodeSatisfied(father_node)) { is_one_father_of_node_satisfied = true; }
      });
      if (is_one_father_of_node_satisfied) { NodeHandler(node); }
    }
  });
}

std::function<bool(OpNode*)> MakePredicatorIsAllowedToRunWithHalf(const OpGraph& op_graph) {
  auto allowed_set = std::make_shared<HashSet<OpNode*>>();
  op_graph.ForEachNode([&](OpNode* node) {
    if (node->parallel_desc().device_type() != DeviceType::kGPU) { return; }
    for (const std::string& obn : node->op().output_bns()) {
      LogicalBlobId lbi = node->op().BnInOp2Lbi(obn);
      // TODO(niuchong): this ain't right for fw-bw-opgraph, but right for fw-opgraph
      if (node->BatchAxis4Lbi(lbi).has_value() == false) { return; }
    }
    INSERT_CHECK(allowed_set->insert(node));
  });
  return [allowed_set](OpNode* node) -> bool { return IsKeyFound(*allowed_set, node); };
}

bool TryUpdtBnVal4SepcialOpConf(const OperatorConf::OpTypeCase& op_type, PbMessage* op_conf,
                                const std::string& old_val, const std::string& new_val,
                                const std::string& dst_ibn) {
  if (OperatorConf::kPrintConf == op_type) {
    const std::pair<std::string, int32_t> prefix_idx = GenUnRepeatedBn(dst_ibn);
    CHECK_EQ("in", prefix_idx.first);
    PbMessage* print_record_conf =
        MutableRepeatedMessageInPbMessage(op_conf, "in", prefix_idx.second);
    ReplaceStrValInPbFdOrPbRpf(print_record_conf, "lbn", old_val, new_val);
    return true;
  }
  return false;
}

std::string ReplaceSlashToDash4Lbn(std::string lbn) {
  std::replace(lbn.begin(), lbn.end(), '/', '-');
  return lbn;
}

void InsertCastOpImpl(bool f2h, const OpGraph& op_graph, const HashSet<OpNode*>& white_set,
                      JobBuilder* job_builder) {
  HashSet<OpEdge*> white_set_edges;
  {
    std::function<const std::unordered_set<OpEdge*>&(OpNode*)> Node2Edges =
        f2h ? &OpNode::in_edges : &OpNode::out_edges;
    std::function<OpNode*(OpEdge*)> OppositeNode = f2h ? &OpEdge::src_node : &OpEdge::dst_node;
    op_graph.ForEachNode([&](OpNode* node) {
      if (IsKeyFound(white_set, node)) {
        for (OpEdge* edge : Node2Edges(node)) {
          if (!IsKeyFound(white_set, OppositeNode(edge))) {
            INSERT_CHECK(white_set_edges.insert(edge));
          }
        }
      }
    });
    auto EdgeName4Edge = [](OpEdge* const& edge) {
      return std::string("edge_of_") + edge->src_node()->op().op_name() + "_to_"
             + edge->dst_node()->op().op_name();
    };
    VLOG(3) << "white_set_edges for f2h value: " << f2h << " is "
            << Container2Str<HashSet<OpEdge*>, OpEdge*>(white_set_edges, EdgeName4Edge);
  }

  HashMap<std::string, std::vector<OpEdge*>> edges_group_by_lbn;
  {
    for (OpEdge* edge : white_set_edges) {
      CHECK_EQ(1, edge->lbis().size());
      std::string lbn = GenLogicalBlobName(edge->lbis().front());
      edges_group_by_lbn[lbn].push_back(edge);
    }
  }

  HashMap<std::string, OperatorConf> dst_op_name2dst_op_confs;
  for (auto& pair : edges_group_by_lbn) {
    const std::string& lbn = pair.first;
    OpNode* src_node = pair.second.front()->src_node();
    OperatorConf cast_op_conf;
    {
      std::string cast_suffix = f2h ? "-cast_f2h" : "-cast_h2f";
      DataType cast_data_type = f2h ? DataType::kFloat16 : DataType::kFloat;
      cast_op_conf.set_name(ReplaceSlashToDash4Lbn(lbn) + cast_suffix);
      CastOpConf* cast_conf = cast_op_conf.mutable_cast_conf();
      cast_conf->set_in(lbn);
      cast_conf->set_out("out");
      cast_conf->set_data_type(cast_data_type);
      job_builder->AddOps(src_node->parallel_desc().parallel_conf(),
                          std::vector<OperatorConf>{cast_op_conf});
      LOG(INFO) << "Insert CastOp: " << cast_op_conf.name() << " between " << lbn;
    }

    std::string new_lbn = cast_op_conf.name() + "/out";
    for (OpEdge* edge : pair.second) {
      CHECK(src_node == edge->src_node());
      OpNode* dst_node = edge->dst_node();
      LogicalBlobId cur_lbi = edge->lbis().front();
      CHECK_EQ(lbn, GenLogicalBlobName(cur_lbi));
      CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
      const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

      const std::string& dst_op_name = dst_node->op().op_name();
      if (!IsKeyFound(dst_op_name2dst_op_confs, dst_op_name)) {
        INSERT_CHECK(
            dst_op_name2dst_op_confs.insert(std::make_pair(dst_op_name, dst_node->op().op_conf())));
      }
      OperatorConf& dst_op_conf = dst_op_name2dst_op_confs.at(dst_op_name);
      PbMessage* dst_op_type_conf =
          MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
      std::string new_lbn = cast_op_conf.name() + "/out";
      if (!TryUpdtBnVal4SepcialOpConf(dst_op_conf.op_type_case(), dst_op_type_conf, lbn, new_lbn,
                                      dst_ibn)) {
        ReplaceInputLbnInOpCustomizedConf(dst_op_type_conf, dst_ibn, lbn, new_lbn);
      }
    }
  }

  std::vector<OperatorConf> dst_op_confs;
  for (const auto& pair : dst_op_name2dst_op_confs) { dst_op_confs.push_back(pair.second); }
  // make sure an op_conf can only be udpated once, cuz later update will override before
  job_builder->MutOpsOnlyOnce(dst_op_confs);
}

class AutoMixedPrecision final : public OpGraphPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoMixedPrecision);
  AutoMixedPrecision()
      : white_list_(AutoMixedPrecisionLists::WhiteList()),
        black_list_(AutoMixedPrecisionLists::BlackList()),
        gray_list_(AutoMixedPrecisionLists::GrayList()),
        clear_list_(AutoMixedPrecisionLists::ClearList()) {}
  ~AutoMixedPrecision() = default;

  bool IsEnabled() const override { return GlobalJobDesc().enable_auto_mixed_precision(); }

  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;

 private:
  void FillBlackSet(const OpGraph& op_graph, HashSet<OpNode*>* black_set) const;
  void FillWhiteSet(const OpGraph& op_graph, std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                    const HashSet<OpNode*>& black_set, HashSet<OpNode*>* white_set) const;
  void PropagateWhiteThroughClearNodes(const OpGraph& op_graph,
                                       std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                                       const HashSet<OpNode*>& black_set,
                                       HashSet<OpNode*>* white_set) const;
  void InsertCastOp(const OpGraph& op_graph, const HashSet<OpNode*>& white_set,
                    JobBuilder* job_builder) const;

  const AMPList& white_list_;
  const AMPList& black_list_;
  const AMPList& gray_list_;
  const AMPList& clear_list_;
};

void AutoMixedPrecision::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  CHECK_GE(CUDA_VERSION, 10000);
  CHECK(GlobalJobDesc().DefaultDataType() == DataType::kFloat);

  std::function<std::string(OpNode* const&)> OpName4Node = [](OpNode* const& node) {
    return node->op().op_name();
  };
  HashSet<OpNode*> black_set;
  HashSet<OpNode*> white_set;

  FillBlackSet(op_graph, &black_set);
  VLOG(1) << "BlackSet include: "
          << Container2Str<HashSet<OpNode*>, OpNode*>(black_set, OpName4Node);

  auto IsAllowedToRunWithHalf = MakePredicatorIsAllowedToRunWithHalf(op_graph);
  FillWhiteSet(op_graph, IsAllowedToRunWithHalf, black_set, &white_set);
  VLOG(2) << "WhiteSet Before Propagate include: "
          << Container2Str<HashSet<OpNode*>, OpNode*>(white_set, OpName4Node);
  PropagateWhiteThroughClearNodes(op_graph, IsAllowedToRunWithHalf, black_set, &white_set);
  VLOG(1) << "WhiteSet include: "
          << Container2Str<HashSet<OpNode*>, OpNode*>(white_set, OpName4Node);

  InsertCastOp(op_graph, white_set, job_builder);
}

void AutoMixedPrecision::FillBlackSet(const OpGraph& op_graph, HashSet<OpNode*>* black_set) const {
  HashSet<OpNode*> upstream_or_part_of_black_and_gray;
  DfsTopoGraphTraversal(
      op_graph, true,
      [&](OpNode* node) {
        return IsNodeInList(black_list_, node) || IsNodeInList(gray_list_, node);
      },
      [&](OpNode* node) { return IsNodeInList(clear_list_, node); },
      [&](OpNode* node) { return IsKeyFound(upstream_or_part_of_black_and_gray, node); },
      [&](OpNode* node) {
        INSERT_CHECK(upstream_or_part_of_black_and_gray.insert(node));
        VLOG(3) << "FillBlackSet(): Insert " << node->op().op_name()
                << " to upstream_or_part_of_black_and_gray";
      });

  // propagate black through upstream_or_part_of_black_and_gray
  DfsTopoGraphTraversal(
      op_graph, false, [&](OpNode* node) { return IsNodeInList(black_list_, node); },
      [&](OpNode* node) { return IsKeyFound(upstream_or_part_of_black_and_gray, node); },
      [&](OpNode* node) { return IsKeyFound(*black_set, node); },
      [&](OpNode* node) {
        INSERT_CHECK(black_set->insert(node));
        VLOG(2) << "FillBlackSet(): Insert " << node->op().op_name() << " to black_set";
      });
}

void AutoMixedPrecision::FillWhiteSet(const OpGraph& op_graph,
                                      std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                                      const HashSet<OpNode*>& black_set,
                                      HashSet<OpNode*>* white_set) const {
  HashSet<OpNode*> upstream_or_part_of_white;
  auto IsWhiteAndAllowedToRunHalf = [&](OpNode* node) {
    return IsAllowedToRunWithHalf(node) && IsNodeInList(white_list_, node);
  };
  DfsTopoGraphTraversal(
      op_graph, true, IsWhiteAndAllowedToRunHalf,
      [&](OpNode* node) {
        return !IsKeyFound(black_set, node) && IsAllowedToRunWithHalf(node)
               && (IsNodeInList(gray_list_, node) || IsNodeInList(clear_list_, node));
      },
      [&](OpNode* node) { return IsKeyFound(upstream_or_part_of_white, node); },
      [&](OpNode* node) {
        INSERT_CHECK(upstream_or_part_of_white.insert(node));
        VLOG(3) << "FillWhiteSet(): Insert " << node->op().op_name()
                << " to upstream_or_part_of_white";
      });

  DfsTopoGraphTraversal(op_graph, false, IsWhiteAndAllowedToRunHalf,
                        [&](OpNode* node) { return IsKeyFound(upstream_or_part_of_white, node); },
                        [&](OpNode* node) { return IsKeyFound(*white_set, node); },
                        [&](OpNode* node) {
                          INSERT_CHECK(white_set->insert(node));
                          VLOG(2) << "FillWhiteSet(): Insert " << node->op().op_name()
                                  << " to white_set";
                        });
}

void AutoMixedPrecision::PropagateWhiteThroughClearNodes(
    const OpGraph& op_graph, std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
    const HashSet<OpNode*>& black_set, HashSet<OpNode*>* white_set) const {
  auto PropagateIntoOneDirection = [&](bool is_downward) {
    DfsTopoGraphTraversal(op_graph, !is_downward, [&](OpNode* node) { return false; },
                          [&](OpNode* node) {
                            return !IsKeyFound(*white_set, node) && !IsKeyFound(black_set, node)
                                   && IsNodeInList(clear_list_, node)
                                   && IsAllowedToRunWithHalf(node);
                          },
                          [&](OpNode* node) { return IsKeyFound(*white_set, node); },
                          [&](OpNode* node) {
                            INSERT_CHECK(white_set->insert(node));
                            VLOG(2) << "PropagateWhiteThroughNonListNodes(): Insert "
                                    << node->op().op_name() << " to white_set";
                          });
  };
  PropagateIntoOneDirection(true);
  PropagateIntoOneDirection(false);
}

void AutoMixedPrecision::InsertCastOp(const OpGraph& op_graph, const HashSet<OpNode*>& white_set,
                                      JobBuilder* job_builder) const {
  InsertCastOpImpl(true, op_graph, white_set, job_builder);
  InsertCastOpImpl(false, op_graph, white_set, job_builder);
}

REGISTER_FUNCTION_PASS("AutoMixedPrecision", AutoMixedPrecision);

}  // namespace

}  // namespace oneflow
