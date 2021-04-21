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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

class PipelineBufferPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PipelineBufferPass);
  PipelineBufferPass() = default;
  ~PipelineBufferPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const std::string kBufferOpNamePrefix = "System-Pipeline-Buffer-Op_";

const Scope& Scope4ScopeSymbolId(int64_t scope_symbol_id) {
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id));
  return Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
}

const Scope& Scope4OpNode(const OpNode* op_node) {
  const OperatorConf& op_conf = op_node->op().op_conf();
  CHECK(op_conf.has_scope_symbol_id());
  return Scope4ScopeSymbolId(op_conf.scope_symbol_id());
}

bool IsForwardPassScope(const Scope& scope) {
  return scope.scope_proto().calculation_pass_name() == kForwardPass;
}

bool IsBackwardPassScope(const Scope& scope) {
  const std::string& calculation_pass_name = scope.scope_proto().calculation_pass_name();
  return calculation_pass_name == kBackwardPass;
}

bool IsIdentityBufferOpNode(const OpNode* node) {
  const OperatorConf& op_conf = node->op().op_conf();
  if (op_conf.has_user_conf()) {
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name == "identity_buffer" || op_type_name == "repeat") { return true; }
    if (op_conf.name().find("loss-") != std::string::npos) { return true; }
  }
  return false;
}

Maybe<void> PipelineBufferPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  int64_t repeat_num = GlobalJobDesc().job_conf().num_gradient_accumulation_steps();
  if (repeat_num <= 1) { return Maybe<void>::Ok(); }

  HashSet<const OpNode*> no_scope_op_node;
  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (!this_node->op().op_conf().has_scope_symbol_id()) {
      no_scope_op_node.insert(this_node);
      LOG(WARNING) << " op : " << this_node->op().op_conf().DebugString() << " has NOT scope!";
    }
  });

  HashMap<std::string, OperatorConf> buffer_op_name2conf;
  HashMap<std::string, OperatorConf> mut_op_name2conf;
  HashMap<std::string, const OpNode*> buffer_op_name2src_op_node;

  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (no_scope_op_node.find(this_node) != no_scope_op_node.end()) {
      return;  // ignore op without scope
    }
    if (!IsBackwardPassScope(Scope4OpNode(this_node))) {
      return;  // ignore fw dst op
    }
    const std::string& this_op_name = this_node->op().op_name();
    for (const OpEdge* in_edge : this_node->in_edges()) {
      const OpNode* src_node = in_edge->src_node();
      if (no_scope_op_node.find(src_node) != no_scope_op_node.end()) {
        continue;  // ignore op without scope
      }
      if (IsForwardPassScope(Scope4OpNode(src_node)) && (!IsIdentityBufferOpNode(src_node))) {
        LOG(WARNING) << "DataEdge: src_op[FwPass]: " << src_node->op().op_conf().DebugString()
                     << " dst_op[BwPass]: " << this_node->op().op_conf().DebugString()
                     << " connected without buffer op.";

        for (const LogicalBlobId& lbi : in_edge->lbis()) {
          std::string lbn = GenLogicalBlobName(lbi);
          std::string buffer_op_name =
              kBufferOpNamePrefix + "-" + lbi.op_name() + "-" + lbi.blob_name();
          auto it = buffer_op_name2conf.find(buffer_op_name);
          if (it == buffer_op_name2conf.end()) {
            auto ret_pair = buffer_op_name2conf.emplace(
                buffer_op_name, user_op::UserOpConfWrapperBuilder(buffer_op_name)
                                    .Op("identity_buffer")
                                    .Input("in", lbn)
                                    .Output("out")
                                    .Attr<int64_t>("buffer_size", repeat_num)
                                    .ScopeSymbolId(src_node->op().op_conf().scope_symbol_id())
                                    .Build()
                                    .op_conf());
            CHECK(buffer_op_name2src_op_node.emplace(buffer_op_name, src_node).second);
            CHECK(ret_pair.second);
            it = ret_pair.first;
          }

          auto mut_op_it = mut_op_name2conf.find(this_op_name);
          if (mut_op_it == mut_op_name2conf.end()) {
            auto ret_pair = mut_op_name2conf.emplace(this_op_name, this_node->op().op_conf());
            CHECK(ret_pair.second);
            mut_op_it = ret_pair.first;
          }

          const std::string buffer_out = user_op::UserOpConfWrapper(it->second).output("out", 0);

          for (const std::string& ibn : in_edge->lbi2ibns().at(lbi)) {
            std::string old_lbn =
                ReplaceInputLbnInOpCustomizedConf(&(mut_op_it->second), ibn, buffer_out);
            CHECK_EQ(old_lbn, lbn);
          }
        }
      }
    }
    for (const std::string& ctrl_in_op_name : this_node->op().op_conf().ctrl_in_op_name()) {
      const OpNode* src_node = op_graph.OpNode4OpName(ctrl_in_op_name);
      if (no_scope_op_node.find(src_node) != no_scope_op_node.end()) {
        continue;  // ignore op without scope
      }
      if (IsForwardPassScope(Scope4OpNode(src_node))) {
        LOG(WARNING) << "CtrlEdge: src_op[FwPass]: " << src_node->op().op_conf().DebugString()
                     << " dst_op[BwPass]: " << this_node->op().op_conf().DebugString()
                     << " connected.";
      }
    }
  });

  for (auto& pair : buffer_op_name2conf) {
    const OperatorConf& conf = pair.second;
    const OpNode* src_node = buffer_op_name2src_op_node.at(pair.first);
    JUST(job_builder->AddOp(src_node->parallel_desc().parallel_conf(), conf));
    LOG(WARNING) << " Add buffer Op: " << conf.DebugString();
  }
  for (auto& pair : mut_op_name2conf) {
    JUST(job_builder->MutOpOnlyOnce(pair.second));
    LOG(WARNING) << " mut op: " << pair.second.DebugString();
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PipelineBufferPass", PipelineBufferPass);

}  // namespace oneflow
