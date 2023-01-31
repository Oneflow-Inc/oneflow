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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/graph/node.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

namespace {

struct RelativeNodes {
  const OpNode* input_node = nullptr;
  const OpNode* output_node = nullptr;
  const OpNode* nearest_del_node = nullptr;
  std::vector<const OpNode*> in_ctrl_nodes = {};
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

bool IsDependOPNodeAtTop(const OpNode* op_node, HashSet<const OpNode*>& del_nodes) {
  CHECK(IsDependyOp(op_node->op().op_conf()));
  const OpNode* input_op_node = GetNodeFromInputEdge(op_node);
  const OpNode* in_ctrl_op_node = GetNodeFromInCtrlEdge(op_node);
  if (del_nodes.find(input_op_node) == del_nodes.end()
      && del_nodes.find(in_ctrl_op_node) == del_nodes.end()) {
    return true;
  } else {
    return false;
  }
}

void GetRelativeNodesHelper(const OpNode* op_node, const HashSet<const OpNode*>& del_nodes,
                            const OpNode* input_node, std::vector<const OpNode*> in_ctrl_nodes,
                            std::vector<RelativeNodes>& ret) {
  CHECK(IsDependyOp(op_node->op().op_conf()));
  for (const OpEdge* out_edge : op_node->out_edges()) {
    const OpNode* out_op_node = out_edge->dst_node();
    if (del_nodes.find(out_op_node) == del_nodes.end()) {
      // "out_op_node" is one of valid output nodes
      // in this case, record the nodes as result and finish the recursion

      // put node connected to in-ctrl edge into "in_ctrl_nodes" if not depend OP 
      const OpNode* in_ctrl_node_to_check = GetNodeFromInCtrlEdge(op_node);
      if (del_nodes.find(in_ctrl_node_to_check) == del_nodes.end()) {
        in_ctrl_nodes.emplace_back(in_ctrl_node_to_check);
      }

      // set node connected to input edge as "in_node" if not depend OP 
      // otherwise remain the value of "in_node"
      const OpNode* input_node_to_check = GetNodeFromInputEdge(op_node);
      if (del_nodes.find(input_node_to_check) == del_nodes.end()) {
        // should not have two input nodes for a depend OP Chain
        CHECK(input_node == nullptr);
        input_node = input_node_to_check;
      }

      ret.push_back({input_node, out_op_node, op_node, in_ctrl_nodes});
    } else if (op_node == GetNodeFromInCtrlEdge(out_op_node)) {
      // "out_op_node" is ALSO a depend OP Node, and "op_node" connect to its in-ctrl edge
      // in this case, all precursor nodes of "op_node" should be seen as in-ctrl OP Node
      
      // put "input_node" into "in_ctrl_nodes" if not NULL 
      if (input_node) in_ctrl_nodes.push_back(input_node);

      // put node connected to in-ctrl edge into "in_ctrl_nodes" if not depend OP 
      const OpNode* in_ctrl_node_to_check = GetNodeFromInCtrlEdge(op_node);
      if (del_nodes.find(in_ctrl_node_to_check) == del_nodes.end()) {
        in_ctrl_nodes.emplace_back(in_ctrl_node_to_check);
      }

      // put node connected to input edge into "in_ctrl_nodes" if not depend OP 
      const OpNode* input_node_to_check = GetNodeFromInputEdge(op_node);
      if (del_nodes.find(input_node_to_check) == del_nodes.end()) {
        in_ctrl_nodes.push_back(input_node_to_check);
      }

      // set "input_node" as NULL in subsequent recursion
      // indicate that real precursor of the target output node has not been found yet
      input_node = nullptr;
      // continue recursion until the target output node is found
      GetRelativeNodesHelper(out_op_node, del_nodes, input_node, in_ctrl_nodes, ret);
    } else {
      // "out_op_node" is ALSO a depend OP Node, and "op_node" connect to its in-ctrl edge

      // in this case, "input_node" should be the real precursor of the target output node
      // thus, remain or update(if NULL) "input_node", then pass it to subsequent processing
      // and append new in-ctrl OP into in_ctrl_nodes

      // put node connected to in-ctrl edge into "in_ctrl_nodes" if not depend OP 
      const OpNode* in_ctrl_node_to_check = GetNodeFromInCtrlEdge(op_node);
      if (del_nodes.find(in_ctrl_node_to_check) == del_nodes.end()) {
        in_ctrl_nodes.emplace_back(in_ctrl_node_to_check);
      }

      // set node connected to input edge as "in_node" if not depend OP
      // otherwise remain the value of "in_node"
      const OpNode* input_node_to_check = GetNodeFromInputEdge(op_node);
      if (del_nodes.find(input_node_to_check) == del_nodes.end()) {
        // should not have two input nodes for a depend OP Chain
        CHECK(input_node == nullptr);
        input_node = input_node_to_check;
      }
      // continue recursion until the target output node is found
      GetRelativeNodesHelper(out_op_node, del_nodes, input_node, in_ctrl_nodes, ret);
    }
  }
}

const std::vector<RelativeNodes> GetRelativeNodes(const OpNode* op_node,
                                                  const HashSet<const OpNode*>& del_nodes) {
  std::vector<RelativeNodes> ret;
  GetRelativeNodesHelper(op_node, del_nodes, nullptr, {}, ret);
  return ret;
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

  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });

  HashSet<const OpNode*> del_nodes;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const std::string& op_name = op_node->op().op_name();
    const OperatorConf& op_conf = op_node->op().op_conf();
    // not depend op
    if (!IsDependyOp(op_conf)) { return; }
    // has ctrl in
    if (!op_conf.ctrl_in_op_name().empty()) { return; }
    // is ctrl in of another op
    if (ctrl_in_op_names.find(op_name) != ctrl_in_op_names.end()) { return; }

    del_nodes.insert(op_node);
  });

  HashMap<std::string, OperatorConf> to_update_op_confs;
  std::vector<std::string> del_op_names;
  del_op_names.reserve(del_nodes.size());
  for (const OpNode* op_node : del_nodes) {
    del_op_names.emplace_back(op_node->op().op_name());
    // GetRelativeNodes() has considered the chain of multiple depend OP Nodes and processes them
    // from top to down, so skip the intermediate nodes
    if (!IsDependOPNodeAtTop(op_node, del_nodes)) { continue; }
    const std::vector<RelativeNodes> relatives = GetRelativeNodes(op_node, del_nodes);

    // adjust op_conf of those nodes related to depend OP nodes
    for (const RelativeNodes& item : relatives) {
      const OpNode* input_node = item.input_node;
      const OpNode* output_node = item.output_node;
      const OpNode* nearest_del_node = item.nearest_del_node;
      const std::vector<const OpNode*>& depend_nodes = item.in_ctrl_nodes;

      const auto& old_lbi = nearest_del_node->op().BnInOp2Lbi(nearest_del_node->op().SoleObn());
      const auto& new_lbi = input_node->op().BnInOp2Lbi(input_node->op().SoleObn());
      const Operator& out_op = output_node->op();
      for (const std::string& ibn : out_op.input_bns()) {
        if (out_op.BnInOp2Lbi(ibn) != old_lbi) { continue; }

        auto iter = to_update_op_confs.find(out_op.op_name());
        if (iter == to_update_op_confs.end()) {
          iter = to_update_op_confs.emplace(out_op.op_name(), out_op.op_conf()).first;
        }
        OperatorConf& out_op_conf = iter->second;
        // connect input_node and output_node
        const auto& old_val =
            ReplaceInputLbnInOpCustomizedConf(&out_op_conf, ibn, GenLogicalBlobName(new_lbi));
        CHECK_EQ_OR_RETURN(GenLogicalBlobName(old_lbi), old_val);

        // add in-ctrl OPs
        for (const OpNode* node : depend_nodes) {
          CHECK(output_node != node);  // self-loop found
          const auto& existed_ctrl_in_op_names = op_node->op().op_conf().ctrl_in_op_name();
          const std::string& new_ctrl_in_op_name = node->op().op_name();
          auto existed_it = std::find(existed_ctrl_in_op_names.begin(),
                                      existed_ctrl_in_op_names.end(), new_ctrl_in_op_name);
          // avoid adding input node or duplicate control nodes
          if (node != input_node && existed_it == existed_ctrl_in_op_names.end()) {
            out_op_conf.add_ctrl_in_op_name(new_ctrl_in_op_name);
          }
        }
      }
    }
  }

  JobBuilder job_builder(job);
  for (const auto& pair : to_update_op_confs) { job_builder.MutOpsOnlyOnce({pair.second}); }
  job_builder.DelOps(del_op_names);

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneDependOpPass", PruneDependOpPass);

}  // namespace oneflow
