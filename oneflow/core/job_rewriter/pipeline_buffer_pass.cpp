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

bool IsForwardPass(const OpNode* node) {
  return Scope4OpNode(node).scope_proto().calculation_pass_name() == kForwardPass;
}

bool IsBackwardPass(const OpNode* node) {
  return Scope4OpNode(node).scope_proto().calculation_pass_name() == kBackwardPass;
}

bool OpNodeHasScope(const OpNode* node) { return node->op().op_conf().has_scope_symbol_id(); }

bool IsIdentityBufferOrRepeatOpNode(const OpNode* node) {
  const OperatorConf& op_conf = node->op().op_conf();
  if (op_conf.has_user_conf()) {
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name == "identity_buffer" || op_type_name == "repeat") { return true; }
  }
  return false;
}

int64_t GetStageIdHint(const OpNode* node) {
  return Scope4OpNode(node).Int64("pipeline_stage_id_hint");
}

void TryInsertOrUseBufferOp(const OpEdge* op_edge, const int64_t buffer_size,
                            HashMap<std::string, OperatorConf>* buffer_op_name2op_conf,
                            HashMap<std::string, ParallelConf>* buffer_op_name2parallel_conf,
                            HashMap<std::string, OperatorConf>* mut_op_name2conf) {
  const OpNode* src_node = op_edge->src_node();
  const OpNode* dst_node = op_edge->dst_node();
  const std::string& dst_op_name = dst_node->op().op_name();
  const int64_t stage_id = GetStageIdHint(dst_node);
  for (const LogicalBlobId& lbi : op_edge->lbis()) {
    std::string lbn = GenLogicalBlobName(lbi);
    std::string buffer_op_name = kBufferOpNamePrefix + "-" + lbi.op_name() + "-" + lbi.blob_name()
                                 + "-stage_id_" + std::to_string(stage_id);
    auto it = buffer_op_name2op_conf->find(buffer_op_name);
    if (it == buffer_op_name2op_conf->end()) {
      it = buffer_op_name2op_conf
               ->emplace(buffer_op_name,
                         user_op::UserOpConfWrapperBuilder(buffer_op_name)
                             .Op("identity_buffer")
                             .Input("in", lbn)
                             .Output("out")
                             .Attr<int64_t>("buffer_size", buffer_size)
                             .ScopeSymbolId(src_node->op().op_conf().scope_symbol_id())
                             .Build()
                             .op_conf())
               .first;
      CHECK(buffer_op_name2parallel_conf
                ->emplace(buffer_op_name, dst_node->parallel_desc().parallel_conf())
                .second);
    }

    auto mut_op_it = mut_op_name2conf->find(dst_op_name);
    if (mut_op_it == mut_op_name2conf->end()) {
      mut_op_it = mut_op_name2conf->emplace(dst_op_name, dst_node->op().op_conf()).first;
    }

    const std::string buffer_out = user_op::UserOpConfWrapper(it->second).output("out", 0);
    for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
      std::string old_lbn =
          ReplaceInputLbnInOpCustomizedConf(&(mut_op_it->second), ibn, buffer_out);
      CHECK_EQ(old_lbn, lbn);
    }

    LOG(INFO) << "\n Insert buffer op: [" << buffer_op_name << "](buffer_size:" << buffer_size
              << ") \n from [" << src_node->op().op_name() << "] -> [" << dst_node->op().op_name()
              << "] \n";
  }
}

Maybe<void> PipelineBufferPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  if (GlobalJobDesc().job_conf().num_gradient_accumulation_steps() <= 1) {
    return Maybe<void>::Ok();
  }

  int64_t max_stage_id = 0;
  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (!OpNodeHasScope(this_node)) {
      LOG(WARNING) << " op : " << this_node->op().op_conf().DebugString() << " has NOT scope!";
    }
    max_stage_id = std::max(max_stage_id, GetStageIdHint(this_node));
  });

  if (max_stage_id == 0) { return Maybe<void>::Ok(); }
  const int64_t total_stage_num = max_stage_id + 1;
  LOG(INFO) << "total stage num = " << total_stage_num;

  HashMap<std::string, OperatorConf> buffer_op_name2op_conf;
  HashMap<std::string, ParallelConf> buffer_op_name2parallel_conf;
  HashMap<std::string, OperatorConf> mut_op_name2conf;

  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (!OpNodeHasScope(this_node)) { return; /* ignore op without scope */ }
    if (!IsBackwardPass(this_node)) { return; /* ignore fw dst op */ }
    for (const OpEdge* in_edge : this_node->in_edges()) {
      const OpNode* src_node = in_edge->src_node();
      if (!OpNodeHasScope(src_node)) { continue; /* ignore op without scope */ }
      const int64_t src_stage_id = GetStageIdHint(src_node);
      const int64_t dst_stage_id = GetStageIdHint(this_node);
      const int64_t buffer_size = total_stage_num - 1 - dst_stage_id;
      CHECK_GE(buffer_size, 0);
      CHECK_LT(buffer_size, total_stage_num);
      if (buffer_size == 0) { continue; /* last stage(loss) does NOT need to insert buffer */ }

      if (IsForwardPass(src_node) && (!IsIdentityBufferOrRepeatOpNode(src_node))) {
        if (src_stage_id != dst_stage_id) {
          LOG(WARNING) << " Cross diff stage link From: [" << src_node->op().op_conf().DebugString()
                       << "](stage_id:" << std::to_string(src_stage_id) << ") -> ["
                       << this_node->op().op_conf().DebugString()
                       << "](stage_id:" << std::to_string(dst_stage_id) << ")\n";
        }
        TryInsertOrUseBufferOp(in_edge, buffer_size, &buffer_op_name2op_conf,
                               &buffer_op_name2parallel_conf, &mut_op_name2conf);
      }
    }
    for (const std::string& ctrl_in_op_name : this_node->op().op_conf().ctrl_in_op_name()) {
      const OpNode* src_node = op_graph.OpNode4OpName(ctrl_in_op_name);
      if (!OpNodeHasScope(src_node)) { continue; /* ignore op without scope */ }
      if (IsForwardPass(src_node)) {
        LOG(WARNING) << "CtrlEdge: src_op[FwPass]: " << src_node->op().op_conf().DebugString()
                     << " dst_op[BwPass]: " << this_node->op().op_conf().DebugString()
                     << " connected.";
      }
    }
  });

  op_graph.ForEachEdge([&](const OpEdge* edge) {
    const OpNode* src_node = edge->src_node();
    const OpNode* dst_node = edge->dst_node();
    if (OpNodeHasScope(src_node) && OpNodeHasScope(dst_node) && IsForwardPass(src_node)
        && IsForwardPass(dst_node)) {
      const int64_t src_stage_id = GetStageIdHint(src_node);
      const int64_t dst_stage_id = GetStageIdHint(dst_node);
      // NOTE(chengcheng): buffer_size = stage diff - 1, because CopyHD can act as a buffer.
      //   Buffer size need be careful in some complex case.
      const int64_t buffer_size = dst_stage_id - src_stage_id - 1;
      if (src_stage_id < dst_stage_id && buffer_size >= 1) {
        TryInsertOrUseBufferOp(edge, buffer_size, &buffer_op_name2op_conf,
                               &buffer_op_name2parallel_conf, &mut_op_name2conf);
      }
    }
  });

  for (auto& pair : buffer_op_name2op_conf) {
    CHECK(buffer_op_name2parallel_conf.find(pair.first) != buffer_op_name2parallel_conf.end());
    JUST(job_builder->AddOp(buffer_op_name2parallel_conf.at(pair.first), pair.second));
  }
  for (auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PipelineBufferPass", PipelineBufferPass);

}  // namespace oneflow
