/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifdef WITH_CUDA

#include "oneflow/core/job_rewriter/auto_mixed_precision_lists.h"

#include <algorithm>

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/pass_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

void VerifyAMPList(const AMPList& amp_list) {
  for (const auto& op_type : amp_list) {
    CHECK(user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type) != nullptr)
        << "Cannot find " << op_type << " of AutoMixedPrecision list in OpRegistry.";
  }
}

using OpArg = std::pair<std::string, int32_t>;
using NoCastRegistry = std::multimap<std::string, OpArg>;

NoCastRegistry* GetNoCastRegistry() {
  static NoCastRegistry s_registry;
  return &s_registry;
}

bool FindInNoCastRegisry(const std::string& op_type, const OpArg& op_arg) {
  auto range = GetNoCastRegistry()->equal_range(op_type);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second == op_arg) { return true; }
  }
  return false;
}

std::function<bool(OpNode*)> MakePredicatorIsAllowedToRunWithHalf(const OpGraph& op_graph) {
  auto allowed_set = std::make_shared<HashSet<OpNode*>>();
  op_graph.ForEachNode([&](OpNode* node) {
    if (node->parallel_desc().device_type() != DeviceType::kGPU) { return; }
    if (node->op().output_bns().size() > 0) { INSERT_CHECK(allowed_set->insert(node)); }
  });
  return [allowed_set](OpNode* node) -> bool { return IsKeyFound(*allowed_set, node); };
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
      return std::string("edge of\t") + edge->src_node()->op().op_name() + "\tto\t"
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

    const BlobDesc& blob_desc = src_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(lbn));
    if (blob_desc.data_type() != DataType::kFloat) { continue; }

    std::string cast_suffix = f2h ? "-cast_f2h" : "-cast_h2f";
    DataType cast_data_type = f2h ? DataType::kFloat16 : DataType::kFloat;
    auto cast_op = user_op::UserOpConfWrapperBuilder(ReplaceSlashToDash4Lbn(lbn) + cast_suffix)
                       .Op("cast")
                       .Input("in", lbn)
                       .Output("out")
                       .Attr<DataType>("dtype", cast_data_type)
                       .ScopeSymbolId(src_node->op().op_conf().scope_symbol_id())
                       .Build();

    bool cast_is_consumed = false;
    for (OpEdge* edge : pair.second) {
      CHECK(src_node == edge->src_node());
      OpNode* dst_node = edge->dst_node();
      LogicalBlobId cur_lbi = edge->lbis().front();
      CHECK_EQ(lbn, GenLogicalBlobName(cur_lbi));
      CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
      const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

      if (dst_node->op().op_conf().has_user_conf()) {
        const std::string& op_type = dst_node->op().op_conf().user_conf().op_type_name();
        const auto& op_arg = GenUnRepeatedBn(dst_ibn);
        if (FindInNoCastRegisry(op_type, op_arg)) { continue; }
      }

      cast_is_consumed = true;

      const std::string& dst_op_name = dst_node->op().op_name();
      if (!IsKeyFound(dst_op_name2dst_op_confs, dst_op_name)) {
        INSERT_CHECK(
            dst_op_name2dst_op_confs.insert(std::make_pair(dst_op_name, dst_node->op().op_conf())));
      }
      OperatorConf& dst_op_conf = dst_op_name2dst_op_confs.at(dst_op_name);
      std::string new_lbn = cast_op.op_name() + "/out_0";
      CHECK_EQ(lbn, ReplaceInputLbnInOpCustomizedConf(&dst_op_conf, dst_ibn, new_lbn));
    }

    if (cast_is_consumed) {
      job_builder->AddOps(src_node->parallel_desc().parallel_conf(),
                          std::vector<OperatorConf>{cast_op.op_conf()});
      LOG(INFO) << "Insert CastOp: " << cast_op.op_name() << " between " << lbn;
    }
  }

  std::vector<OperatorConf> dst_op_confs;
  for (const auto& pair : dst_op_name2dst_op_confs) { dst_op_confs.push_back(pair.second); }
  // make sure an op_conf can only be udpated once, cuz later update will override before
  job_builder->MutOpsOnlyOnce(dst_op_confs);
}

class AutoMixedPrecision final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoMixedPrecision);
  AutoMixedPrecision()
      : white_list_(AutoMixedPrecisionLists::WhiteList()),
        black_list_(AutoMixedPrecisionLists::BlackList()),
        gray_list_(AutoMixedPrecisionLists::GrayList()),
        clear_list_(AutoMixedPrecisionLists::ClearList()) {}
  ~AutoMixedPrecision() = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().enable_auto_mixed_precision();
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

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

Maybe<void> AutoMixedPrecision::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  CHECK_GE(CUDA_VERSION, 10000);
  CHECK(GlobalJobDesc().DefaultDataType() == DataType::kFloat);

  VerifyAMPList(white_list_);
  VerifyAMPList(black_list_);
  VerifyAMPList(gray_list_);
  VerifyAMPList(clear_list_);

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
  return Maybe<void>::Ok();
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

REGISTER_JOB_PASS("AutoMixedPrecision", AutoMixedPrecision);

}  // namespace

namespace {

struct NoCastRegistrar final {
  NoCastRegistrar(const std::string& op_type, OpArg&& op_arg) {
    auto* registry = GetNoCastRegistry();
    registry->emplace(std::make_pair(op_type, std::move(op_arg)));
  }
  ~NoCastRegistrar() = default;
};

#define REGISTER_NO_CAST_REGISTRY(op_type, input_arg_name, idx)       \
  static NoCastRegistrar OF_PP_CAT(g_registrar, __COUNTER__)(op_type, \
                                                             std::make_pair(input_arg_name, idx));

// For Example:
// REGISTER_NO_CAST_REGISTRY("matmul", "b", 0);

REGISTER_NO_CAST_REGISTRY("normalization", "moving_mean", 0)
REGISTER_NO_CAST_REGISTRY("normalization", "moving_variance", 0)
REGISTER_NO_CAST_REGISTRY("normalization", "gamma", 0)
REGISTER_NO_CAST_REGISTRY("normalization", "beta", 0)

REGISTER_NO_CAST_REGISTRY("normalization_add_relu", "moving_mean", 0)
REGISTER_NO_CAST_REGISTRY("normalization_add_relu", "moving_variance", 0)
REGISTER_NO_CAST_REGISTRY("normalization_add_relu", "gamma", 0)
REGISTER_NO_CAST_REGISTRY("normalization_add_relu", "beta", 0)

}  // namespace

}  // namespace oneflow

#endif  // WITH_CUDA
