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
#include <glog/logging.h>
#include <string>
#include <vector>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/graph/node.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

namespace {

struct UpdatedNodeInfo {
  const OpNode* node = nullptr;
  const OpNode* new_src_node = nullptr;
  const OpNode* depend_node_nearest_src = nullptr;
  const OpNode* depend_node_nearest_dst = nullptr;
  std::vector<const OpNode*> new_in_ctrl_nodes;
  bool updated = false;
};

bool IsDependyOp(const OperatorConf& op) {
  return op.has_user_conf() && (op.user_conf().op_type_name() == "depend");
}

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), IsDependyOp);
}

const OpNode* GetNodeFromEdgeByTensorName(const OpNode* op_node,
                                          const std::string& target_tensor_name) {
  CHECK(IsDependyOp(op_node->op().op_conf()));
  for (const OpEdge* in_edge : op_node->in_edges()) {
    const OpNode* in_op_node = in_edge->src_node();
    const std::string& in_op_node_name = in_op_node->op().op_name();
    const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns = in_edge->lbi2ibns();

    for (const auto& item : lbi2ibns) {
      const std::string& lbi_op_name = item.first.op_name();
      for (const std::string& tensor_name : item.second) {
        if (in_op_node_name == lbi_op_name && tensor_name == target_tensor_name) {
          return in_op_node;
        }
      }
    }
  }
  return nullptr;
}

const OpNode* GetNodeFromInputEdge(const OpNode* op_node) {
  return GetNodeFromEdgeByTensorName(op_node, "in_0");
}

const OpNode* GetNodeFromInCtrlEdge(const OpNode* op_node) {
  return GetNodeFromEdgeByTensorName(op_node, "depend_tensor_0");
}

LogicalBlobId GetNewLbi(const OpNode* src_node, const OpNode* depend_node_nearest_src) {
  CHECK(IsDependyOp(depend_node_nearest_src->op().op_conf()));
  for (const OpEdge* out_edge : src_node->out_edges()) {
    const OpNode* dst_node = out_edge->dst_node();
    if (dst_node != depend_node_nearest_src) { continue; }

    CHECK(out_edge->lbis().size() == 1);
    return out_edge->lbis()[0];
  }
  // should not reach here
  CHECK(false);
  return {};
}

class PruneDependOpPass final : public JobPass {
 public:
  PruneDependOpPass() = default;
  ~PruneDependOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> PruneDependOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ctx->job_desc().prune_depend_ops()) { return Maybe<void>::Ok(); }
  if (!NeedDoPass(*job)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);

  HashMap<std::string, UpdatedNodeInfo> node_info_with_update;
  std::vector<const OpNode*> ordered_nodes;

  // Step 0: topological sort, setup a map for recording modification
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    UpdatedNodeInfo node_info;
    node_info.node = node;
    node_info_with_update.emplace(node->op().op_name(), node_info);
    ordered_nodes.emplace_back(node);
  });

  // Step 1: process node by topological order
  // record modification info when meet Depend OP nodes
  for (const OpNode* cur_node : ordered_nodes) {
    const std::string& cur_op_name = cur_node->op().op_name();
    const OperatorConf& cur_op_conf = cur_node->op().op_conf();
    if (!IsDependyOp(cur_op_conf)) { continue; }

    // record modification info to each dst_node
    for (const OpEdge* out_edge : cur_node->out_edges()) {
      const OpNode* dst_node = out_edge->dst_node();
      const Operator& dst_op = dst_node->op();

      UpdatedNodeInfo& updated_dst_node_info = node_info_with_update.find(dst_op.op_name())->second;
      UpdatedNodeInfo& updated_cur_node_info = node_info_with_update.find(cur_op_name)->second;
      updated_dst_node_info.updated = true;
      updated_dst_node_info.depend_node_nearest_dst = cur_node;

      // Step 1.1: record a new in-ctrl node
      const OpNode* cur_in_ctrl_node = GetNodeFromInCtrlEdge(cur_node);
      updated_dst_node_info.new_in_ctrl_nodes.emplace_back(cur_in_ctrl_node);

      // Step 1.2: inherit in-ctrl nodes from Depend OP nodes
      const auto& ori_in_ctrl_op_names = cur_op_conf.ctrl_in_op_name();
      for (const std::string& ori_ctrl_in_op_name : ori_in_ctrl_op_names) {
        updated_dst_node_info.new_in_ctrl_nodes.emplace_back(
            node_info_with_update[ori_ctrl_in_op_name].node);
      }
      if (updated_cur_node_info.updated) {
        std::vector<const OpNode*>& inherit_in_ctrl_nodes = updated_cur_node_info.new_in_ctrl_nodes;
        for (const OpNode* inherit_in_ctrl_node : inherit_in_ctrl_nodes) {
          updated_dst_node_info.new_in_ctrl_nodes.emplace_back(inherit_in_ctrl_node);
        }
      }

      // Step 1.3 process src nodes
      const OpNode* cur_src_node = GetNodeFromInputEdge(cur_node);
      if (IsDependyOp(dst_node->op().op_conf()) && cur_node == GetNodeFromInCtrlEdge(dst_node)) {
        // "cur_node" and "dst_node" are all Depend OP nodes, and their connection is like this
        // other_node   cur_node
        //          \   /
        //         dst_node
        // in this case, all src nodes of "cur_node" should be seen as in-ctrl nodes
        if (updated_cur_node_info.updated && updated_cur_node_info.new_src_node) {
          updated_dst_node_info.new_in_ctrl_nodes.emplace_back(updated_cur_node_info.new_src_node);
        }
        updated_dst_node_info.new_in_ctrl_nodes.emplace_back(cur_src_node);
      } else {
        if (!IsDependyOp(cur_src_node->op().op_conf())) {
          updated_dst_node_info.new_src_node = cur_src_node;
          updated_dst_node_info.depend_node_nearest_src = cur_node;
        } else if (updated_cur_node_info.updated && updated_cur_node_info.new_src_node) {
          updated_dst_node_info.new_src_node = updated_cur_node_info.new_src_node;
          updated_dst_node_info.depend_node_nearest_src =
              updated_cur_node_info.depend_node_nearest_src;
        }
      }
    }
  }

  // Step 2: extract modification info
  // including new connection and to delete nodes
  std::vector<std::string> del_node_names;
  HashMap<std::string, OperatorConf> to_update_op_confs;
  for (const auto& node_info : node_info_with_update) {
    // filter nodes not updated
    if (!node_info.second.updated) { continue; }
    const OpNode* cur_node = node_info.second.node;
    const std::string& cur_op_name = cur_node->op().op_name();
    // filter Depnd nodes
    if (IsDependyOp(cur_node->op().op_conf())) {
      del_node_names.emplace_back(cur_op_name);
      continue;
    }

    const Operator& cur_op = cur_node->op();
    auto iter = to_update_op_confs.find(node_info.first);
    if (iter == to_update_op_confs.end()) {
      iter = to_update_op_confs.emplace(node_info.first, cur_op.op_conf()).first;
    }
    OperatorConf& cur_op_conf = iter->second;

    // Step 2.1: connect updated src_node with cur_node (dst_node of Depned OP)
    const OpNode* src_node = node_info.second.new_src_node;
    const OpNode* depend_node_nearest_dst = node_info.second.depend_node_nearest_dst;
    const OpNode* depend_node_nearest_src = node_info.second.depend_node_nearest_src;
    CHECK(src_node && depend_node_nearest_dst && depend_node_nearest_src);
    const auto& old_lbi =
        depend_node_nearest_dst->op().BnInOp2Lbi(depend_node_nearest_dst->op().SoleObn());
    const auto new_lbi = GetNewLbi(src_node, depend_node_nearest_src);
    for (const std::string& ibn : cur_node->op().input_bns()) {
      if (cur_op.BnInOp2Lbi(ibn) == old_lbi) {
        const auto& old_val =
            ReplaceInputLbnInOpCustomizedConf(&cur_op_conf, ibn, GenLogicalBlobName(new_lbi));
        CHECK_EQ(GenLogicalBlobName(old_lbi), old_val);
        VLOG(3) << "Update input edge, Src Node: " << src_node->op().op_name()
                << "\t->\tDst Node: " << cur_op_name;
      }
    }

    // Step 2.2: add in-ctrl OPs
    const auto& existed_ctrl_in_op_names = cur_op_conf.ctrl_in_op_name();
    for (const OpNode* in_ctrl_node : node_info.second.new_in_ctrl_nodes) {
      // filter Depnd nodes
      if (IsDependyOp(in_ctrl_node->op().op_conf())) { continue; }
      CHECK(cur_node != in_ctrl_node);  // self-loop found
      const std::string& new_ctrl_in_op_name = in_ctrl_node->op().op_name();
      auto existed_it = std::find(existed_ctrl_in_op_names.begin(), existed_ctrl_in_op_names.end(),
                                  new_ctrl_in_op_name);
      // filter src node or duplicate in-ctrl nodes
      if (in_ctrl_node != src_node && existed_it == existed_ctrl_in_op_names.end()) {
        cur_op_conf.add_ctrl_in_op_name(new_ctrl_in_op_name);
        VLOG(3) << "Add in-ctrl edge, Src Node: " << new_ctrl_in_op_name
                << "\t->\tDst Node: " << cur_op_name;
      }
    }
  }

  // Step 3: apply modification to job
  JobBuilder job_builder(job);
  for (const auto& pair : to_update_op_confs) { job_builder.MutOpsOnlyOnce({pair.second}); }
  job_builder.DelOps(del_node_names);
  return Maybe<void>::Ok();
};

}  // namespace

REGISTER_JOB_PASS("PruneDependOpPass", PruneDependOpPass);

}  // namespace oneflow
