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
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/user/ops/nccl_logical_util.h"

namespace oneflow {

namespace {

class NcclLogicalOpFusionPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalOpFusionPass);
  NcclLogicalOpFusionPass() = default;
  ~NcclLogicalOpFusionPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
           && EnableNcclLogicalFusion();
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const std::string kNcclLogicalFusionOpNamePrefix = "Sys-NCCL-Logical-Fusion";

bool IsNcclLogicalOpNode(const OpNode* node) {
  if (node->op().op_conf().has_user_conf()) {
    const std::string& user_type_name = node->op().op_conf().user_conf().op_type_name();
    if (user_type_name == "_nccl_logical_all_reduce"
        || user_type_name == "_nccl_logical_reduce_scatter"
        || user_type_name == "_nccl_logical_reduce_scatter_noncontinuous"
        || user_type_name == "_nccl_logical_all_gather"
        || user_type_name == "_nccl_logical_all_gather_noncontinuous"
        || user_type_name == "_nccl_logical_s2s"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_reduce"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_gather"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous"
        || user_type_name == "_nccl_logical_2D_same_dim0_all2all"
        || user_type_name == "_nccl_logical_2D_same_dim1_all_reduce"
        /* || user_type_name == "_nccl_logical_send_recv" */) {
      // TODO(chengcheng) : support nccl send/recv kernel
      return true;
    }
  }
  return false;
}

Maybe<void> ReplaceNcclOpsWithFusionOp(std::vector<OperatorConf>* nccl_fusion_ops,
                                       std::vector<ParallelConf>* nccl_fusion_op_parallel_confs,
                                       std::unordered_set<std::string>* del_ops,
                                       HashMap<std::string, OperatorConf>* mut_op_name2conf,
                                       const std::vector<const OpNode*>& nccl_ops) {
  if (nccl_ops.size() <= 1) { return Maybe<void>::Ok(); }
  const int32_t nccl_size = nccl_ops.size();
  const OpNode* first_nccl = nccl_ops.front();
  const OperatorConf& first_nccl_conf = first_nccl->op().op_conf();
  const ParallelDesc& seed_placement = first_nccl->parallel_desc();
  const int64_t scope_symbol_id = first_nccl_conf.scope_symbol_id();
  std::vector<std::string> src_nd_sbp_str_list;
  std::vector<std::string> dst_nd_sbp_str_list;
  std::vector<std::string> nccl_type_list;
  int64_t logical_chain_id = first_nccl_conf.logical_chain_id();
  bool has_stream_name_hint = first_nccl_conf.has_stream_name_hint();
  std::string stream_name_hint = first_nccl_conf.stream_name_hint();
  user_op::UserOpConfWrapperBuilder fusion_builder =
      user_op::UserOpConfWrapperBuilder("Sys-NCCL-fusion-" + NewUniqueId());
  fusion_builder.OpTypeName("_nccl_logical_fusion");
  for (const OpNode* nccl_op : nccl_ops) {
    fusion_builder.Input("in",
                         GenLogicalBlobName(nccl_op->op().BnInOp2Lbi(nccl_op->op().SoleIbn())));
    src_nd_sbp_str_list.push_back(
        NdSbpToLongString(nccl_op->NdSbp4BnInOp(nccl_op->op().SoleIbn())));
    dst_nd_sbp_str_list.push_back(
        NdSbpToLongString(nccl_op->NdSbp4BnInOp(nccl_op->op().SoleObn())));
    nccl_type_list.push_back(nccl_op->op().op_conf().user_conf().op_type_name());
    CHECK_OR_RETURN(seed_placement == nccl_op->parallel_desc());
    CHECK_EQ_OR_RETURN(has_stream_name_hint, nccl_op->op().op_conf().has_stream_name_hint());
    CHECK_EQ_OR_RETURN(stream_name_hint, nccl_op->op().op_conf().stream_name_hint());
    // 1. update del op
    VLOG(3) << " Del op: " << nccl_op->op().op_conf().DebugString();
    del_ops->insert(nccl_op->op().op_name());
  }

  auto fusion_nccl_op =
      fusion_builder.Output("out", nccl_size)
          .Attr<std::vector<std::string>>("src_nd_sbp_str_list", src_nd_sbp_str_list)
          .Attr<std::vector<std::string>>("dst_nd_sbp_str_list", dst_nd_sbp_str_list)
          .Attr<std::vector<std::string>>("nccl_type_list", nccl_type_list)
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  OperatorConf fusion_nccl_op_conf = fusion_nccl_op.op_conf();
  fusion_nccl_op_conf.set_logical_chain_id(logical_chain_id);
  if (has_stream_name_hint) { fusion_nccl_op_conf.set_stream_name_hint(stream_name_hint); }

  // 2. update fusion op
  VLOG(3) << " Add fusion op : " << fusion_nccl_op_conf.DebugString()
          << " \n with placement: " << seed_placement.parallel_conf().DebugString();
  nccl_fusion_ops->push_back(fusion_nccl_op_conf);
  nccl_fusion_op_parallel_confs->push_back(seed_placement.parallel_conf());

  for (int32_t i = 0; i < nccl_size; ++i) {
    std::string output_lbn = fusion_nccl_op.output("out", i);
    std::string input_lbn = fusion_nccl_op.input("in", i);
    const OpNode* origin_nccl = JUST(VectorAt(nccl_ops, i));
    const OpEdge* origin_edge = origin_nccl->SoleOutEdge();
    std::string origin_nccl_input_lbn =
        GenLogicalBlobName(origin_nccl->op().BnInOp2Lbi(origin_nccl->op().SoleIbn()));
    std::string origin_nccl_output_lbn =
        GenLogicalBlobName(origin_nccl->op().BnInOp2Lbi(origin_nccl->op().SoleObn()));
    CHECK_EQ_OR_RETURN(input_lbn, origin_nccl_input_lbn);
    const OpNode* origin_consumer = origin_edge->dst_node();
    const std::string& consumer_op_name = origin_consumer->op().op_name();
    if (mut_op_name2conf->find(consumer_op_name) == mut_op_name2conf->end()) {
      mut_op_name2conf->emplace(consumer_op_name, origin_consumer->op().op_conf());
    }
    CHECK_EQ_OR_RETURN(origin_edge->lbis().size(), 1);
    const LogicalBlobId& lbi = origin_edge->lbis().front();
    VLOG(3) << " input_lbn: " << input_lbn;
    VLOG(3) << " lbi: " << GenLogicalBlobName(lbi);
    CHECK_EQ_OR_RETURN(origin_nccl_output_lbn, GenLogicalBlobName(lbi));

    // 3. update consumer op
    for (const std::string& ibn : JUST(MapAt(origin_edge->lbi2ibns(), lbi))) {
      std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
          &JUST(MapAt(*mut_op_name2conf, consumer_op_name)), ibn, output_lbn);
      CHECK_EQ_OR_RETURN(old_lbn, origin_nccl_output_lbn);
    }

    VLOG(3) << " Update origin consumer op from: \n [ "
            << origin_consumer->op().op_conf().DebugString() << " ] \n to \n [ "
            << JUST(MapAt(*mut_op_name2conf, consumer_op_name)).DebugString() << " ] \n";
  }
  return Maybe<void>::Ok();
}

Maybe<void> NcclLogicalOpFusionPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  HashMap<const OpNode*, int64_t> op_node2nccl_depth;
  HashMap<int64_t, std::vector<const OpNode*>> nccl_depth2nccl_ops;
  auto ConstForEachDataAndCtrlInNode = [&](const OpNode* node,
                                           const std::function<void(const OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge(Handler);
    for (const auto& ctrl_in_op_name : node->op().op_conf().ctrl_in_op_name()) {
      const OpNode* in_node = op_graph.OpNode4OpName(ctrl_in_op_name);
      CHECK(in_node) << " cannot find ctrl_in_op_name: [" << ctrl_in_op_name << "] of op: ["
                     << node->op().op_name() << "] in OpGraph. ";
      Handler(in_node);
    }
  };
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    int64_t nccl_depth = 0;
    ConstForEachDataAndCtrlInNode(node, [&](const OpNode* in_node) {
      auto it = op_node2nccl_depth.find(in_node);
      CHECK(it != op_node2nccl_depth.end());  // topo search
      nccl_depth = std::max(nccl_depth, it->second);
    });
    if (IsNcclLogicalOpNode(node)) {
      nccl_depth++;  // ONLY nccl node update depth
      nccl_depth2nccl_ops[nccl_depth].push_back(node);
    }
    CHECK(op_node2nccl_depth.emplace(node, nccl_depth).second);
  });

  if (nccl_depth2nccl_ops.empty()) { return Maybe<void>::Ok(); }

  std::vector<OperatorConf> nccl_fusion_ops;
  std::vector<ParallelConf> nccl_fusion_op_parallel_confs;

  std::unordered_set<std::string> del_ops;
  HashMap<std::string, OperatorConf> mut_op_name2conf;

  for (const auto& pair : nccl_depth2nccl_ops) {
    HashMap<int64_t, HashMap<std::string, std::vector<const OpNode*>>> chain2comm2nccl_ops;
    for (const OpNode* nccl_op : pair.second) {
      CHECK_OR_RETURN(nccl_op->op().op_conf().has_logical_chain_id());
      int64_t logical_chain_id = nccl_op->op().op_conf().logical_chain_id();
      const auto& hierarchy = nccl_op->parallel_desc().hierarchy();
      std::string comm_key =
          hierarchy->ToString()
          + GetCommKeyFromNcclType(nccl_op->op().op_conf().user_conf().op_type_name());
      LOG(INFO) << "ccdebuglog: comm_key = " << comm_key << " nccl_op: " << nccl_op->op().op_name();
      chain2comm2nccl_ops[logical_chain_id][comm_key].push_back(nccl_op);
    }
    for (const auto& chain_pair : chain2comm2nccl_ops) {
      for (const auto& comm_pair : chain_pair.second) {
        JUST(ReplaceNcclOpsWithFusionOp(&nccl_fusion_ops, &nccl_fusion_op_parallel_confs, &del_ops,
                                        &mut_op_name2conf, comm_pair.second));
      }
    }
  }

  job_builder->RemoveOpByName(del_ops);
  for (const auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }
  CHECK_EQ_OR_RETURN(nccl_fusion_ops.size(), nccl_fusion_op_parallel_confs.size());
  for (int32_t i = 0; i < nccl_fusion_ops.size(); ++i) {
    JUST(job_builder->AddOp(JUST(VectorAt(nccl_fusion_op_parallel_confs, i)),
                            JUST(VectorAt(nccl_fusion_ops, i))));
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("NcclLogicalOpFusionPass", NcclLogicalOpFusionPass);

}  // namespace oneflow

#endif  // WITH_CUDA
