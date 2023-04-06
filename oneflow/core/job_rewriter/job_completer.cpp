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
#include "oneflow/core/job_rewriter/job_completer.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job_rewriter/autotick.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job_rewriter/group_boxing_by_dst_parallel.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job_rewriter/boxing_with_middle_nodes.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/common/cost_util.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

namespace {

Maybe<void> CheckOpGraph(const OpGraph& op_graph) {
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    size_t in_cnt = 0;
    op_graph.ForEachDataAndCtrlInNode(op_node, [&](OpNode*) { ++in_cnt; });
    if (in_cnt == 0) { CHECK_OR_RETURN(op_node->op().op_conf().has_wait_and_send_ids_conf()); }

    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });

    if (out_cnt == 0) { CHECK_OR_RETURN(op_node->op().op_conf().has_callback_notify_conf()); }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> CheckAndLogOpGraph(const Job& job) {
  auto op_graph = std::make_unique<OpGraph>(job);
  // Check op graph.
  JUST(CheckOpGraph(*op_graph));
  // Log op graph.
  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    const JobDesc& job_desc = GlobalJobDesc();
    TeePersistentLogStream::Create(StrCat("optimized_job", job_desc.job_id()))->Write(job);
    op_graph->ToDotWithFilePath("optimized_dlnet_" + std::to_string(job_desc.job_id())
                                + "_op_graph.dot");
  }
  return Maybe<void>::Ok();
}

Maybe<void> WithOpGraphAndMutJob(Job* job,
                                 const std::function<Maybe<void>(const OpGraph&, Job*)>& Handler) {
  OpGraph op_graph(*job);
  JUST(Handler(op_graph, job));
  return Maybe<void>::Ok();
}

Maybe<void> WithOpGraphAndMutJobBuilder(
    Job* job, const std::function<Maybe<void>(const OpGraph&, JobBuilder*)>& Handler) {
  OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  JUST(Handler(op_graph, &job_builder));
  return Maybe<void>::Ok();
}

Maybe<void> SetCtrlInOpName4VariableOp(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto IsMutableConsumedLbi = [](const Operator& op, const LogicalBlobId& lbi) -> bool {
    for (const std::string& bn : op.input_bns()) {
      if (op.BnInOp2Lbi(bn) == lbi && op.InputBlobModifier4Ibn(bn).is_mutable()) { return true; }
    }
    return false;
  };
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  HashMap<const OperatorConf*, HashSet<std::string>> op_conf2ctrl_in_op_names;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
    if (op_node->out_edges().size() <= 1) { return Maybe<void>::Ok(); }
    const Operator& variable_op = op_node->op();
    const LogicalBlobId& variable_lbi = variable_op.BnInOp2Lbi(variable_op.SoleObn());
    const OperatorConf* mutable_consumer = nullptr;
    std::vector<const OperatorConf*> naive_consumers;
    naive_consumers.reserve(op_node->out_edges().size());
    for (OpEdge* edge : op_node->out_edges()) {
      const auto& op_conf = edge->dst_node()->op().op_conf();
      if (IsMutableConsumedLbi(edge->dst_node()->op(), variable_lbi)) {
        CHECK_OR_RETURN(mutable_consumer == nullptr);
        mutable_consumer = &op_conf;
      } else {
        naive_consumers.emplace_back(&op_conf);
      }
    }
    if (mutable_consumer == nullptr) { return Maybe<void>::Ok(); }
    for (const auto* fw_bw_op : naive_consumers) {
      op_conf2ctrl_in_op_names[mutable_consumer].insert(fw_bw_op->name());
    }
    return Maybe<void>::Ok();
  }));
  for (const auto& pair : op_conf2ctrl_in_op_names) {
    OperatorConf mut_mutable_consumer_op_conf(*pair.first);
    for (const auto& fw_bw_op_name : pair.second) {
      if (!IsReachable(fw_bw_op_name, mut_mutable_consumer_op_conf.name())) {
        mut_mutable_consumer_op_conf.add_ctrl_in_op_name(fw_bw_op_name);
      }
    }
    JUST(job_builder->MutOpOnlyOnce(mut_mutable_consumer_op_conf));
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> JobCompleter::Complete(Job* job) {
  const auto& job_name = job->job_conf().job_name();
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  // NOTE(chengcheng): disable this pass for reduce boxing memory life cycle to memory cost.
  auto compile_tc = std::make_unique<CostCounter<std::chrono::seconds>>(true, true);
  if (!Singleton<ResourceDesc, ForSession>::Get()
           ->resource()
           .disable_group_boxing_by_dst_parallel()) {
    JUST(WithOpGraphAndMutJobBuilder(job, &GroupBoxingByDstParallel));
  }
  compile_tc->Count("[GraphCompile]" + job_name + " GroupBoxingByDstParallel", 1, true);
  if (GlobalProcessCtx::WorldSize() > 1) {
    JUST(WithOpGraphAndMutJobBuilder(job, &BoxingWithMiddleNodes));
  }
  compile_tc->Count("[GraphCompile]" + job_name + " BoxingWithMiddleNodes", 1, true);
  JUST(WithOpGraphAndMutJobBuilder(job, &SetCtrlInOpName4VariableOp));
  compile_tc->Count("[GraphCompile]" + job_name + " SetCtrl", 1, true);
  // complete tick ops
  JUST(WithOpGraphAndMutJobBuilder(job, &AutoPrependTick));
  compile_tc->Count("[GraphCompile]" + job_name + " AutoPrependTick", 1, true);
  JUST(WithOpGraphAndMutJobBuilder(job, &AddTickForTimeShape));
  compile_tc->Count("[GraphCompile]" + job_name + " AddTickForTimeShape", 1, true);
  JUST(WithOpGraphAndMutJob(job, &MultiClientAutoSourceAndSinkTick));
  compile_tc->Count("[GraphCompile]" + job_name + " AutoSourceAndSinkTick", 1, true);
  JUST(WithOpGraphAndMutJob(job, &MultiClientAutoInterfaceCriticalSectionTick));
  compile_tc->Count("[GraphCompile]" + job_name + " CriticalSectionTick", 1, true);
  JUST(JobPass4Name("SystemOpFillJobNamePass")(job, &job_pass_ctx));
  compile_tc->Count("[GraphCompile]" + job_name + " SystemOpFillJobNamePass", 1, true);
  JUST(JobPass4Name("DumpBlobParallelConfPass")(job, &job_pass_ctx));
  compile_tc->Count("[GraphCompile]" + job_name + " DumpBlobParallelConfPass", 1, true);
#ifdef WITH_CUDA
  if (Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    // NOTE(chengcheng): this pass need as last pass for insert correct op with nccl boxing.
    JUST(JobPass4Name("InsertNcclLogicalOpPass")(job, &job_pass_ctx));
    compile_tc->Count("[GraphCompile]" + job_name + " InsertNcclLogicalOpPass", 1, true);
    // NOTE(chengcheng): Because insert new logical nccl op, MUST dump time shape, sbp again.
    JUST(JobPass4Name("DumpBlobParallelConfPass")(job, &job_pass_ctx));
    compile_tc->Count("[GraphCompile]" + job_name + " DumpBlobParallelConfPass", 1, true);
  }
#endif  // WITH_CUDA
  JUST(JobPass4Name("LogicalChainPass")(job, &job_pass_ctx));
  JUST(JobPass4Name("DumpBlobParallelConfPass")(job, &job_pass_ctx));

  JUST(CheckAndLogOpGraph(*job));
  compile_tc->Count("[GraphCompile]" + job_name + " CheckAndLogOpGraph", 1, true);
  return Maybe<void>::Ok();
}

Maybe<void> JobCompleter::UpdateSharedGraphForNewInput(
    Job* job,
    const std::function<Maybe<std::shared_ptr<one::Tensor>>(const std::string&)>& InputTensor4Name,
    const std::function<Maybe<const OperatorConf*>(const std::string& shared_op_name)>&
        NewOp4SharedOpName) {
  // job is a copy from a shared graph.
  // The job name has already update in py nn.Graph.
  const auto& new_job_name = job->job_conf().job_name();

  const auto& UpdateInputShape = [&InputTensor4Name](OperatorConf& op_conf) -> Maybe<void> {
    // Input op needs to be updated with new input tensor.
    if (op_conf.has_input_conf()) {
      InputOpConf* input_conf = op_conf.mutable_input_conf();
      InterfaceBlobConf* blob_conf = input_conf->mutable_blob_conf();
      auto input_tensor = *JUST(InputTensor4Name(op_conf.name()));
      input_tensor->shape()->ToProto(blob_conf->mutable_shape());
      blob_conf->set_data_type(input_tensor->dtype()->data_type());
    }
    return Maybe<void>::Ok();
  };

  const auto& UpdateAttr = [&NewOp4SharedOpName](OperatorConf& op_conf) -> Maybe<void> {
    // Some op attributes need to be updated with the new traced graph.
    if (op_conf.has_user_conf()) {
      for (auto& pair : *op_conf.mutable_user_conf()->mutable_attr()) {
        if (pair.second.has_at_shape()) {
          const auto* new_op_conf = JUST(NewOp4SharedOpName(op_conf.name()));
          CHECK_EQ_OR_RETURN(new_op_conf->user_conf().op_type_name(),
                             op_conf.user_conf().op_type_name())
              << " new op " << new_op_conf->DebugString() << " is not corresponding with "
              << op_conf.DebugString();
          auto attr_iter = new_op_conf->user_conf().attr().find(pair.first);
          CHECK_OR_RETURN(attr_iter != new_op_conf->user_conf().attr().end())
              << " There is not attr " << pair.first << " in new op " << new_op_conf->DebugString();
          *pair.second.mutable_at_shape() = attr_iter->second.at_shape();
        }
      }
    }
    return Maybe<void>::Ok();
  };

  const auto& UpdateBufferName = [&new_job_name](OperatorConf& op_conf) -> Maybe<void> {
  // These operators' execution depends on new job name.
#define UPDATE_JOB_NAME(op_conf_name)                             \
  if (op_conf.has_##op_conf_name()) {                             \
    op_conf.mutable_##op_conf_name()->set_job_name(new_job_name); \
  }
    UPDATE_JOB_NAME(input_conf);
    UPDATE_JOB_NAME(output_conf);
    UPDATE_JOB_NAME(callback_notify_conf);
    UPDATE_JOB_NAME(wait_and_send_ids_conf);
    UPDATE_JOB_NAME(return_conf);
#undef UPDATE_JOB_NAME

    // Critical section operators depend job_name related buffer_name.
    if (op_conf.has_critical_section_wait_tick_conf()) {
      const auto& buffer_name = op_conf.critical_section_wait_tick_conf().buffer_name();
      if (buffer_name.rfind(kInputCriticalSectionWaitBufferNamePrefix, 0) == 0) {
        op_conf.mutable_critical_section_wait_tick_conf()->set_buffer_name(
            GetInputCriticalSectionWaitBufferName(new_job_name));
      } else if (buffer_name.rfind(kOutputCriticalSectionWaitBufferNamePrefix, 0) == 0) {
        op_conf.mutable_critical_section_wait_tick_conf()->set_buffer_name(
            GetOutputCriticalSectionWaitBufferName(new_job_name));
      }
    }
    if (op_conf.has_critical_section_callback_tick_conf()) {
      const auto& buffer_name = op_conf.critical_section_callback_tick_conf().buffer_name();
      if (buffer_name.rfind(kInputCriticalSectionCallbackBufferNamePrefix, 0) == 0) {
        op_conf.mutable_critical_section_callback_tick_conf()->set_buffer_name(
            GetInputCriticalSectionCallbackBufferName(new_job_name));
      } else if (buffer_name.rfind(kOutputCriticalSectionCallbackBufferNamePrefix, 0) == 0) {
        op_conf.mutable_critical_section_callback_tick_conf()->set_buffer_name(
            GetOutputCriticalSectionCallbackBufferName(new_job_name));
      }
    }
    return Maybe<void>::Ok();
  };

  // Update the job for new input.
  for (auto& op_conf : *job->mutable_net()->mutable_op()) {
    JUST(UpdateInputShape(op_conf));
    JUST(UpdateAttr(op_conf));
    JUST(UpdateBufferName(op_conf));
  }
  // Use OpGraph init to infer all LogicalBlobDesc with the new input shape.
  auto op_graph = std::make_unique<OpGraph>(*job);
  op_graph->DumpLogicalBlobDesc(job);

#ifdef WITH_CUTLASS
  // Warmup cutlass conv with new input shape.
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  JUST(JobPass4Name("CutlassConvTuningWarmupPass")(job, &job_pass_ctx));
#endif  // WITH_CUTLASS

  JUST(CheckAndLogOpGraph(*job));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
