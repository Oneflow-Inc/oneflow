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

namespace oneflow {

namespace {

struct RelativeNodes {
  const OpNode* input = nullptr;
  const OpNode* output = nullptr;
  std::vector<const OpNode*> depends = {};
};

bool IsDependyOp(const OperatorConf& op) {
  return op.has_user_conf() && (op.user_conf().op_type_name() == "depend");
}

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), IsDependyOp);
}

const OpNode* GetNodeFromEdgeByTensorName(const OpNode* op_node,
                                          const std::string target_tensor_name) {
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

const OpNode* GetNodeFromDependEdge(const OpNode* op_node) {
  return GetNodeFromEdgeByTensorName(op_node, "depend_tensor_0");
}

const OpNode* GetValidInputNode(const OpNode* op_node, HashSet<const OpNode*>& del_nodes) {
  CHECK(IsDependyOp(op_node->op().op_conf()));
  const OpNode* input_op_node = GetNodeFromInputEdge(op_node);
  if (del_nodes.find(input_op_node) == del_nodes.end()) { return input_op_node; }
  return nullptr;
}

void GetRelativeNodesHelper(const OpNode* op_node, const HashSet<const OpNode*>& del_nodes,
                            const OpNode* input_node, std::vector<const OpNode*> depend_nodes,
                            std::vector<RelativeNodes>& ret) {
  CHECK(IsDependyOp(op_node->op().op_conf()));
  for (const OpEdge* out_edge : op_node->out_edges()) {
    const OpNode* out_op_node = out_edge->dst_node();
    if (del_nodes.find(out_op_node) == del_nodes.end()) {
      // "out_op_node" is one of valid output nodes

      // in this case, record the nodes as result
      const OpNode* depend_node = GetNodeFromDependEdge(op_node);
      depend_nodes.emplace_back(depend_node);
      ret.push_back({input_node, out_op_node, depend_nodes});
    } else if (op_node == GetNodeFromDependEdge(out_op_node)) {
      // "out_op_node" is ALSO a depend OP Node, and "op_node" is an in-control OP

      // in this case, two precursor node of op_node be interpreted as in-control OP
      // the input_node should not NOT be the precursor of the target output node
      // thus, put input_node into "depend_nodes", and set "input_node" as NULL in
      // subsequent processing
      if (input_node) depend_nodes.push_back(input_node);
      input_node = nullptr;
      // continue recursion until the target output node is found
      GetRelativeNodesHelper(out_op_node, del_nodes, input_node, depend_nodes, ret);
    } else {
      // "out_op_node" is ALSO a depend OP Node, and "op_node" is an input OP

      // in this case, "input_node" should be the real precursor of the target output node
      // thus, append in-ctrl op-node conneted to "op_node" input_node into "depend_nodes",
      // and remain "input_node as" in subsequent processing
      const OpNode* depend_node = GetNodeFromDependEdge(op_node);
      depend_nodes.emplace_back(depend_node);
      // continue recursion until the target output node is found
      GetRelativeNodesHelper(out_op_node, del_nodes, input_node, depend_nodes, ret);
    }
  }
}

const std::vector<RelativeNodes> GetRelativeNodes(const OpNode* op_node,
                                                  const HashSet<const OpNode*>& del_nodes,
                                                  const OpNode* input_node) {
  std::vector<RelativeNodes> ret;
  GetRelativeNodesHelper(op_node, del_nodes, input_node, {}, ret);
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
    // valid input node is node which is bind to 'in' edger of depend OP, and not in del_nodes
    const OpNode* valid_input_node = GetValidInputNode(op_node, del_nodes);
    // GetRelativeNodes() considers the chain of multiple depend OP Nodes and processes them
    // from the beginning, so skip the intermediate nodes whose valid_input_node is NULL
    if (valid_input_node == nullptr) { continue; }
    const std::vector<RelativeNodes> relatives =
        GetRelativeNodes(op_node, del_nodes, valid_input_node);
    const auto& old_lbi = op_node->op().BnInOp2Lbi(op_node->op().SoleObn());

    // adjust op_conf of nodes connected to depend OP Nodes
    for (const RelativeNodes& item : relatives) {
      const OpNode* input_node = item.input;
      const OpNode* output_node = item.output;
      const std::vector<const OpNode*>& depend_nodes = item.depends;
      // in some cases (e.g. the second branch in GetRelativeNodesHelper()), input nodes should
      // be interpreted as control nodes for those case, accordingly their input_node is NULL
      // and the ibn modifications should be skip
      if (input_node) {
        const auto& new_lbi = input_node->op().BnInOp2Lbi(input_node->op().SoleObn());
        const Operator& op = output_node->op();
        for (const std::string& ibn : op.input_bns()) {
          if (op.BnInOp2Lbi(ibn) == old_lbi) {
            auto iter = to_update_op_confs.find(op.op_name());
            if (iter == to_update_op_confs.end()) {
              iter = to_update_op_confs.emplace(op.op_name(), op.op_conf()).first;
            }
            OperatorConf& op_conf = iter->second;
            const auto& old_val =
                ReplaceInputLbnInOpCustomizedConf(&op_conf, ibn, GenLogicalBlobName(new_lbi));
            CHECK_EQ_OR_RETURN(GenLogicalBlobName(old_lbi), old_val);
          }
        }
      }
      // add ctrl_in_op
      const Operator& out_op = output_node->op();
      auto out_iter = to_update_op_confs.find(out_op.op_name());
      if (out_iter == to_update_op_confs.end()) {
        out_iter = to_update_op_confs.emplace(out_op.op_name(), out_op.op_conf()).first;
      }
      OperatorConf& out_op_conf = out_iter->second;
      for (const OpNode* node : depend_nodes) {
        CHECK(output_node != node);  // self-loop found
        const auto& existed_ctrl_in_op_names = op_node->op().op_conf().ctrl_in_op_name();
        const std::string& new_ctrl_in_op_name = node->op().op_name();
        auto existed_it = std::find(existed_ctrl_in_op_names.begin(),
                                    existed_ctrl_in_op_names.end(), new_ctrl_in_op_name);
        if (existed_it == existed_ctrl_in_op_names.end()) {  // avoid adding duplicate control nodes
          out_op_conf.add_ctrl_in_op_name(new_ctrl_in_op_name);
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
