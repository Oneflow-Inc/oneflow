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
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job_rewriter/clone_grad.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/register/op_blob_arg.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/dynamic_loss_scale_job_pass_state.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/job_rewriter/clip_by_global_norm_job_pass_state.h"

namespace oneflow {

namespace {

const TrainConf& GetTrainConf() { return GlobalJobDesc().job_conf().train_conf(); }

int64_t ScopeSymbolId4Lbi(const OpGraph& op_graph, const LogicalBlobId& lbi) {
  return op_graph.OpNode4OpName(lbi.op_name())->op().op_conf().scope_symbol_id();
}

bool AnyLbiWithDiffLbi(const OpEdge* op_edge) {
  const Operator& src_op = op_edge->src_node()->op();
  const Operator& dst_op = op_edge->dst_node()->op();
  auto IsOutputBlobModifierRequiresGrad = [&](const LogicalBlobId& lbi) {
    return src_op.OutputBlobModifier4Obn(op_edge->lbi2obn().at(lbi)).requires_grad();
  };
  auto IsInputBlobModifierRequiresGrad = [&](const LogicalBlobId& lbi) {
    const auto& ibns = op_edge->lbi2ibns().at(lbi);
    for (const std::string& ibn : ibns) {
      if (dst_op.InputBlobModifier4Ibn(ibn).requires_grad()) { return true; }
    }
    CHECK_GT(ibns.size(), 0);
    return false;
  };
  for (const LogicalBlobId& lbi : op_edge->lbis()) {
    if (IsOutputBlobModifierRequiresGrad(lbi) && IsInputBlobModifierRequiresGrad(lbi)) {
      return true;
    }
  }
  CHECK_GT(op_edge->lbis().size(), 0);
  return false;
}

void CheckNotReachableAmongOpNodes(const OpGraph& op_graph, const std::list<OpNode*>& op_nodes) {
  auto IsReachable = op_graph.MakePredicatorIsReachable();
  for (OpNode* src_node : op_nodes) {
    for (OpNode* dst_node : op_nodes) {
      if (src_node == dst_node) { continue; }
      CHECK(!IsReachable(src_node, dst_node));
    }
  }
}

Maybe<void> GetLossOpNodes(const OpGraph& op_graph, std::list<OpNode*>* loss_op_nodes) {
  const auto& train_conf = GetTrainConf();
  HashSet<std::string> loss_op_names;
  for (const std::string& loss_lbn : train_conf.loss_lbn()) {
    loss_op_names.emplace(GenLogicalBlobId(loss_lbn).op_name());
  }
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (loss_op_names.find(op_node->op().op_name()) != loss_op_names.end()) {
      loss_op_nodes->emplace_back(op_node);
    }
  });
  if (loss_op_nodes->empty()) { return Error::LossBlobNotFoundError() << "Loss blob not found."; }
  return Maybe<void>::Ok();
}

Maybe<void> GetLossOpNodesAndAscendants(const OpGraph& op_graph, HashSet<OpNode*>* op_nodes) {
  std::list<OpNode*> starts;
  JUST(GetLossOpNodes(op_graph, &starts));
  auto ForEachNextNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    for (OpEdge* edge : op_node->in_edges()) {
      if (AnyLbiWithDiffLbi(edge)) { Handler(edge->src_node()); }
    }
  };
  op_graph.BfsForEachNode(starts, ForEachNextNode,
                          [&](OpNode* op_node) { op_nodes->emplace(op_node); });
  return Maybe<void>::Ok();
}

std::function<bool(const LogicalBlobId&, const std::string&)> MakePredicatorHasDiff4LbiOpName(
    const OpGraph& op_graph, const std::function<bool(OpNode*)>& NeedBackwardOp) {
  auto lbis2ops_with_in_diff = std::make_shared<HashMap<LogicalBlobId, HashSet<std::string>>>();
  op_graph.ForEachEdge([&](OpEdge* edge) {
    if (NeedBackwardOp(edge->src_node()) && NeedBackwardOp(edge->dst_node())) {
      for (const auto& lbi : edge->lbis()) {
        const auto& obn = edge->lbi2obn().at(lbi);
        if (edge->src_node()->op().OutputBlobModifier4Obn(obn).requires_grad()) {
          (*lbis2ops_with_in_diff)[lbi].emplace(edge->dst_node()->op().op_name());
        }
      }
    }
  });
  return [lbis2ops_with_in_diff](const LogicalBlobId& lbi, const std::string& op_name) {
    if (lbis2ops_with_in_diff->find(lbi) == lbis2ops_with_in_diff->end()) { return false; }
    const auto& op_names = lbis2ops_with_in_diff->at(lbi);
    return op_names.find(op_name) != op_names.end();
  };
}

void GenerateOriginDiffLbi(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                           const LogicalBlobId& lbi, const ParallelDesc& parallel_desc,
                           const std::function<void(const OperatorConf& op_conf)>& AddOp,
                           LogicalBlobId* out_diff_lbi) {
  const TrainConf& train_conf = ctx->job_desc().job_conf().train_conf();
  OperatorConf constant_like_op{};
  constant_like_op.set_name(lbi.op_name() + "_" + lbi.blob_name() + "_grad_ConstantLike");
  ConstantLikeOpConf* constant_like_conf = constant_like_op.mutable_constant_like_conf();
  constant_like_conf->set_like(GenLogicalBlobName(lbi));
  constant_like_conf->set_out("out");
  {
    float origin_grad;
    if (train_conf.has_loss_scale_factor()) {
      origin_grad = train_conf.loss_scale_factor();
    } else {
      origin_grad = 1.0;
    }
    constant_like_conf->set_float_operand(origin_grad);
  }
  AddOp(constant_like_op);
  NdSbp broadcast_nd_sbp;
  for (int32_t i = 0; i < parallel_desc.hierarchy()->NumAxes(); ++i) {
    broadcast_nd_sbp.add_sbp_parallel()->mutable_broadcast_parallel();
  }
  if (train_conf.has_dynamic_loss_scale_policy()) {
    const auto& dynamic_loss_scale_state =
        CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    std::string loss_scale_val_lbn;
    const DataType data_type = op_graph.GetLogicalBlobDesc(lbi).data_type();
    if (data_type == DataType::kFloat) {
      loss_scale_val_lbn = dynamic_loss_scale_state.loss_scale_val_lbn();
    } else {
      auto cast_op =
          user_op::UserOpConfWrapperBuilder(lbi.op_name() + "_" + lbi.blob_name() + "_grad_Cast")
              .Op("cast")
              .Input("in", dynamic_loss_scale_state.loss_scale_val_lbn())
              .Output("out")
              .Attr<DataType>("dtype", data_type)
              .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
              .Build();
      AddOp(cast_op.op_conf());
      OpBlobArg cast_in_op_blob_arg;
      cast_in_op_blob_arg.set_op_name(cast_op.op_name());
      cast_in_op_blob_arg.set_bn_in_op(GenRepeatedBn("in", 0));
      job_builder->SetNdSbp4Oba(cast_in_op_blob_arg, broadcast_nd_sbp);
      OpBlobArg cast_out_op_blob_arg;
      cast_out_op_blob_arg.set_op_name(cast_op.op_name());
      cast_out_op_blob_arg.set_bn_in_op(GenRepeatedBn("out", 0));
      job_builder->SetNdSbp4Oba(cast_out_op_blob_arg, broadcast_nd_sbp);
      loss_scale_val_lbn = cast_op.output("out", 0);
    }
    {
      const OpNode* loss_node = op_graph.OpNode4OpName(lbi.op_name());
      const int64_t time_shape_elem_cnt =
          CHECK_JUST(loss_node->op().GetInputBlobFastestTimeShape())->elem_cnt();
      if (time_shape_elem_cnt != 1) {
        const auto repeat_op = user_op::UserOpConfWrapperBuilder(lbi.op_name() + "_"
                                                                 + lbi.blob_name() + "_grad_Repeat")
                                   .OpTypeName("repeat")
                                   .Input("in", loss_scale_val_lbn)
                                   .Output("out")
                                   .Attr<int32_t>("repeat_num", time_shape_elem_cnt)
                                   .Build();
        AddOp(repeat_op.op_conf());
        OpBlobArg repeat_in_op_blob_arg;
        repeat_in_op_blob_arg.set_op_name(repeat_op.op_name());
        repeat_in_op_blob_arg.set_bn_in_op(GenRepeatedBn("in", 0));
        job_builder->SetNdSbp4Oba(repeat_in_op_blob_arg, broadcast_nd_sbp);
        OpBlobArg repeat_out_op_blob_arg;
        repeat_out_op_blob_arg.set_op_name(repeat_op.op_name());
        repeat_out_op_blob_arg.set_bn_in_op(GenRepeatedBn("out", 0));
        job_builder->SetNdSbp4Oba(repeat_out_op_blob_arg, broadcast_nd_sbp);
        loss_scale_val_lbn = repeat_op.output("out", 0);
      }
    }
    auto scalar_mul_op =
        user_op::UserOpConfWrapperBuilder(lbi.op_name() + "_" + lbi.blob_name() + "_grad_Scale")
            .Op("scalar_mul_by_tensor")
            .Input("x", GenLogicalBlobName(constant_like_op.name(), constant_like_conf->out()))
            .Input("scalar", loss_scale_val_lbn)
            .Output("y")
            .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
            .Build();
    AddOp(scalar_mul_op.op_conf());
    *out_diff_lbi = GenLogicalBlobId(scalar_mul_op.output("y", 0));
  } else {
    out_diff_lbi->set_op_name(constant_like_op.name());
    out_diff_lbi->set_blob_name(constant_like_conf->out());
  }
}

const ParallelConf& ProducerParallelConf4Lbi(const OpGraph& op_graph, const LogicalBlobId& lbi) {
  return op_graph.OpNode4OpName(lbi.op_name())->parallel_desc().parallel_conf();
}

void ScaleModelDiffByConstantLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder,
                                             HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi,
                                             const int64_t loss_instance_num) {
  if (loss_instance_num == 1) { return; }
  const float scale_factor = 1.0f / static_cast<float>(loss_instance_num);
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    auto scalar_mul_op =
        user_op::UserOpConfWrapperBuilder("System-ModelDiffScale-ScalarMul_" + NewUniqueId())
            .Op("scalar_mul")
            .Input("in", GenLogicalBlobName(diff_lbi))
            .Output("out")
            .Attr<bool>("has_float_operand", true)
            .Attr<double>("float_operand", scale_factor)
            .Attr<bool>("has_int_operand", false)
            .Attr<int64_t>("int_operand", 0)
            .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
            .Build();
    job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {scalar_mul_op.op_conf()});
    diff_lbi = GenLogicalBlobId(scalar_mul_op.output("out", 0));
  }
}

Maybe<void> TryLocalCastTotalLossInstanceNum(
    JobBuilder* job_builder, const HashMap<LogicalBlobId, OpNode*>& loss_lbi2loss_node,
    LogicalBlobId* total_loss_instance_num_lbi) {
  auto IsLocal4Lbi = [](const LogicalBlobId& lbi, OpNode* op_node) -> Maybe<bool> {
    const auto& obn = *JUST(op_node->op().obn4lbi(lbi));
    const auto& opt_local_parallel = *JUST(op_node->op().OptLocalParallel4BnInOp(obn));
    return opt_local_parallel.has_local_parallel();
  };
  const auto& begin = *loss_lbi2loss_node.begin();
  bool is_local = JUST(IsLocal4Lbi(begin.first, begin.second));
  for (const auto& pair : loss_lbi2loss_node) {
    bool is_other_local = JUST(IsLocal4Lbi(pair.first, pair.second));
    CHECK_EQ_OR_RETURN(is_local, is_other_local);  // NOLINT
  }
  if (is_local) {
    OperatorConf op_conf;
    op_conf.set_name("System-Cast-Local-TotalLossInstanceNum" + NewUniqueId());
    CastFromLocalOpConf* cast_from_local = op_conf.mutable_cast_from_local_conf();
    cast_from_local->set_in(GenLogicalBlobName(*total_loss_instance_num_lbi));
    cast_from_local->set_out("out");
    cast_from_local->mutable_sbp_parallel()->mutable_partial_sum_parallel();
    const auto& parallel_conf = JUST(job_builder->ParallelConf4Lbi(*total_loss_instance_num_lbi));
    int64_t scope_symbol_id = 0;
    {
      const auto& opt_scope_symbol_id = JUST(MakeInitialScope(job_builder->job().job_conf(),
                                                              SymbolOf(ParallelDesc(parallel_conf)),
                                                              /* is_local */ false))
                                            ->symbol_id();
      CHECK_OR_RETURN(opt_scope_symbol_id.has_value())
          << Error::RuntimeError() << "symbol_id not initialized";
      scope_symbol_id = JUST(opt_scope_symbol_id);
    }
    op_conf.set_scope_symbol_id(scope_symbol_id);
    job_builder->AddOps(parallel_conf, {op_conf});
    total_loss_instance_num_lbi->set_op_name(op_conf.name());
    total_loss_instance_num_lbi->set_blob_name("out");
  }
  return Maybe<void>::Ok();
}

void ScaleModelDiffByDynamicLossInstanceNum(
    const OpGraph& op_graph, JobBuilder* job_builder,
    HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi,
    const HashMap<LogicalBlobId, OpNode*>& loss_lbi2loss_node) {
  auto BuildInstanceNumOpConf4LossOpNode = [&](const LogicalBlobId& loss_lbi, const OpNode* op_node,
                                               LogicalBlobId* lbi) {
    OperatorConf instance_num_op;
    instance_num_op.set_name("System-Autograd-" + loss_lbi.op_name() + "-" + loss_lbi.blob_name()
                             + "-LossInstanceNum");
    auto* instance_num_op_conf = instance_num_op.mutable_shape_elem_cnt_conf();
    instance_num_op_conf->set_x(GenLogicalBlobName(loss_lbi));
    instance_num_op_conf->set_y("y");
    instance_num_op_conf->set_data_type(op_node->LogicalBlobDesc4Lbi(loss_lbi).data_type());
    instance_num_op_conf->mutable_include_axis_conf();
    instance_num_op.set_scope_symbol_id(op_node->op().op_conf().scope_symbol_id());
    job_builder->AddOps(op_node->parallel_desc().parallel_conf(), {instance_num_op});
    lbi->set_op_name(instance_num_op.name());
    lbi->set_blob_name("y");
  };
  LogicalBlobId total_loss_instance_num_lbi;
  if (loss_lbi2loss_node.size() == 1) {
    const auto& pair_it = loss_lbi2loss_node.begin();
    BuildInstanceNumOpConf4LossOpNode(pair_it->first, pair_it->second,
                                      &total_loss_instance_num_lbi);
  } else if (loss_lbi2loss_node.size() > 1) {
    OperatorConf op_conf;
    op_conf.set_name("System-Autograd-total_loss_instance_num");
    TotalLossInstanceNumOpConf* total_loss_instance_num_conf =
        op_conf.mutable_total_loss_instance_num_conf();
    for (const auto& pair : loss_lbi2loss_node) {
      LogicalBlobId loss_instance_num_lbi;
      BuildInstanceNumOpConf4LossOpNode(pair.first, pair.second, &loss_instance_num_lbi);
      total_loss_instance_num_conf->add_in(GenLogicalBlobName(loss_instance_num_lbi));
    }
    total_loss_instance_num_conf->set_out("out");

    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0");
    int64_t scope_symbol_id = 0;
    {
      const auto& opt_scope_symbol_id =
          CHECK_JUST(MakeInitialScope(job_builder->job().job_conf(),
                                      SymbolOf(ParallelDesc(parallel_conf)),
                                      /* is_local */ false))
              ->symbol_id();
      if (!opt_scope_symbol_id.has_value()) { THROW(RuntimeError) << "symbol_id not initialized"; }
      scope_symbol_id = CHECK_JUST(opt_scope_symbol_id);
    }
    op_conf.set_scope_symbol_id(scope_symbol_id);
    job_builder->AddOps(parallel_conf, {op_conf});

    total_loss_instance_num_lbi.set_op_name(op_conf.name());
    total_loss_instance_num_lbi.set_blob_name("out");
  } else {
    UNIMPLEMENTED();
  }
  CHECK_JUST(TryLocalCastTotalLossInstanceNum(job_builder, loss_lbi2loss_node,
                                              &total_loss_instance_num_lbi));
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    auto scalar_div_op =
        user_op::UserOpConfWrapperBuilder("System-ModelDiffScale-ScalarDiv_" + NewUniqueId())
            .Op("scalar_div_by_tensor")
            .Input("x", GenLogicalBlobName(diff_lbi))
            .Input("scalar", GenLogicalBlobName(total_loss_instance_num_lbi))
            .Output("y")
            .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
            .Build();
    job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {scalar_div_op.op_conf()});
    diff_lbi = GenLogicalBlobId(scalar_div_op.output("y", 0));
  }
}

bool AllSplitDistribution(const NdSbp& nd_sbp) {
  for (int64_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
    if (!nd_sbp.sbp_parallel(i).has_split_parallel()) { return false; }
  }
  return true;
}

void InitOutOba2OutDiffLbi(JobPassCtx* ctx, const OpGraph& op_graph,
                           const std::list<OpNode*>& loss_nodes,
                           HashMap<OpBlobArg, LogicalBlobId>* out_oba2out_diff_lbi,
                           JobBuilder* job_builder) {
  for (const std::string& loss_lbn : ctx->job_desc().job_conf().train_conf().loss_lbn()) {
    const LogicalBlobId loss_lbi = GenLogicalBlobId(loss_lbn);
    const auto loss_node_it = std::find_if(
        loss_nodes.cbegin(), loss_nodes.cend(),
        [&](const OpNode* node) { return node->op().op_name() == loss_lbi.op_name(); });
    CHECK(loss_node_it != loss_nodes.cend());
    const OpNode* loss_op_node = *loss_node_it;
    const auto bn_it = std::find_if(
        loss_op_node->op().output_bns().cbegin(), loss_op_node->op().output_bns().cend(),
        [&](const std::string& obn) { return loss_lbi == loss_op_node->op().BnInOp2Lbi(obn); });
    CHECK(bn_it != loss_op_node->op().output_bns().cend());
    LogicalBlobId* out_diff_lbi =
        &(*out_oba2out_diff_lbi)[GenOpBlobArg(loss_op_node->op().op_name(), *bn_it)];
    int64_t scope_symbol_id = loss_op_node->op().op_conf().scope_symbol_id();
    const auto AddOp = [&](const OperatorConf& op_conf) {
      OperatorConf new_op_conf = op_conf;
      new_op_conf.set_scope_symbol_id(scope_symbol_id);
      job_builder->AddOps(loss_op_node->parallel_desc().parallel_conf(), {new_op_conf});
    };
    GenerateOriginDiffLbi(ctx, op_graph, job_builder, loss_lbi, loss_op_node->parallel_desc(),
                          AddOp, out_diff_lbi);
  }
}

void CalcOutLbi2OutDiffLbi(const OpGraph& op_graph,
                           const HashMap<OpBlobArg, LogicalBlobId>& out_oba2out_diff_lbi,
                           HashMap<LogicalBlobId, LogicalBlobId>* out_lbi2out_diff_lbi) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      const auto& lbi = op_node->op().BnInOp2Lbi(obn);
      const auto& oba = GenOpBlobArg(op_node->op().op_name(), obn);
      const auto& out_diff_lbi_it = out_oba2out_diff_lbi.find(oba);
      if (out_diff_lbi_it != out_oba2out_diff_lbi.end()) {
        CHECK(out_lbi2out_diff_lbi->emplace(lbi, out_diff_lbi_it->second).second);
      }
    }
  });
}

void ForEachAggregatedParamGroup(
    const OpGraph& op_graph, const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
    const std::function<void(const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                             const std::vector<LogicalBlobId>& libs)>& Handler) {
  HashMap<LogicalBlobId, const ParallelDesc*> lbi2parallel_desc;
  HashMap<std::pair<ParallelDesc, NdSbp>, std::vector<LogicalBlobId>> group;
  for (auto& pair : lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    const ParallelDesc& parallel_desc = model_op_node->parallel_desc();
    const NdSbp& nd_sbp = model_op_node->NdSbp4Lbi(lbi);
    group[std::make_pair(parallel_desc, nd_sbp)].emplace_back(lbi);
  }
  for (const auto& pair : group) { Handler(pair.first.first, pair.first.second, pair.second); }
}

int64_t MakeScopeSymbolId(const JobConfigProto& job_conf, const ParallelConf& parallel_conf) {
  const auto& opt_scope_symbol_id =
      CHECK_JUST(MakeInitialScope(job_conf, SymbolOf(ParallelDesc(parallel_conf)),
                                  /* is_local */ false))
          ->symbol_id();
  if (!opt_scope_symbol_id.has_value()) { THROW(RuntimeError) << "symbol_id not initialized"; }
  return CHECK_JUST(opt_scope_symbol_id);
}

std::string AddLbns(JobBuilder* job_builder, const std::vector<std::string>& lbns,
                    const ParallelConf& parallel_conf, int64_t scope_symbol_id,
                    const std::string& op_name_prefix) {
  if (lbns.size() == 1) {
    return lbns.front();
  } else {
    user_op::UserOpConfWrapperBuilder add_op_builder(op_name_prefix + NewUniqueId());
    add_op_builder.Op("add_n");
    for (const std::string& lbn : lbns) { add_op_builder.Input("in", lbn); }
    const auto add_op = add_op_builder.Output("out").ScopeSymbolId(scope_symbol_id).Build();
    job_builder->AddOps(parallel_conf, {add_op.op_conf()});
    return add_op.output("out", 0);
  }
}

std::string AddParallelCast(JobBuilder* job_builder, const std::string& in_lbn,
                            const std::string& sbp_str, const ParallelConf& parallel_conf,
                            const std::string& op_name_prefix) {
  ParallelConf flat_parallel_conf = parallel_conf;
  flat_parallel_conf.mutable_hierarchy()->clear_dim();
  const int64_t scope_symbol_id =
      MakeScopeSymbolId(job_builder->job().job_conf(), flat_parallel_conf);
  std::vector<std::string> sbp = {sbp_str};
  auto parallel_cast_op =
      user_op::UserOpConfWrapperBuilder(op_name_prefix + NewUniqueId())
          .Op("hierarchical_parallel_cast")
          .Input("in", in_lbn)
          .Output("out")
          .Attr<std::vector<std::string>>("nd_sbp", sbp)
          .Attr<std::string>("grad_mode", "auto")
          .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>{})
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(flat_parallel_conf, {parallel_cast_op.op_conf()});
  return parallel_cast_op.output("out", 0);
}

bool IsBroadcast(const NdSbp& nd_sbp, const ParallelDesc& parallel_desc) {
  if (parallel_desc.parallel_num() == 1) { return true; }
  for (int64_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
    if (!nd_sbp.sbp_parallel(i).has_broadcast_parallel()) { return false; }
  }
  return true;
}

bool HasSplit(const NdSbp& nd_sbp, const ParallelDesc& parallel_desc) {
  if (parallel_desc.parallel_num() == 1) { return false; }
  for (const auto& sbp : nd_sbp.sbp_parallel()) {
    if (sbp.has_split_parallel()) { return true; }
  }
  return false;
}

OperatorConf GenConstantLikeOp(const std::string& op_name, int64_t scope_symbol_id,
                               const std::string& like_lbn, double value, DataType dtype) {
  OperatorConf op_conf;
  op_conf.set_name(op_name);
  op_conf.set_scope_symbol_id(scope_symbol_id);
  ConstantLikeOpConf* constant_like_conf = op_conf.mutable_constant_like_conf();
  constant_like_conf->set_like(like_lbn);
  if (dtype == DataType::kInt32) {
    constant_like_conf->set_int_operand(static_cast<int32_t>(value));
  } else if (dtype == DataType::kInt64) {
    constant_like_conf->set_int_operand(static_cast<int64_t>(value));
  } else if (dtype == DataType::kFloat) {
    constant_like_conf->set_float_operand(static_cast<float>(value));
  } else if (dtype == DataType::kDouble) {
    constant_like_conf->set_float_operand(value);
  } else {
    UNIMPLEMENTED();
  }
  constant_like_conf->set_data_type(dtype);
  constant_like_conf->set_out("out");
  return op_conf;
}

std::string GlobalAbsMaxMin(const OpGraph& op_graph, JobBuilder* job_builder,
                            const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
                            bool max_or_min, ParallelConf* out_parallel_conf) {
  // max(abs(x))
  bool all_same_parallel_desc = true;
  const ParallelDesc& any_parallel_desc =
      op_graph.OpNode4OpName(lbi2diff_lbi.begin()->first.op_name())->parallel_desc();
  std::vector<std::string> group_reduce_lbns;

  auto GroupReduce = [&](const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                         const std::vector<LogicalBlobId>& lbis) {
    if (!parallel_desc.EqualsIgnoringHierarchy(any_parallel_desc)) {
      all_same_parallel_desc = false;
    }
    int64_t scope_symbol_id =
        MakeScopeSymbolId(job_builder->job().job_conf(), parallel_desc.parallel_conf());
    bool has_split = HasSplit(nd_sbp, parallel_desc);
    if (job_builder->job().job_conf().enable_gradients_stats_aggregation()) {
      std::string multi_reduce_op_type_name =
          has_split ? (max_or_min ? "local_multi_reduce_max_abs" : "local_multi_reduce_min_abs")
                    : (max_or_min ? "multi_reduce_max_abs" : "multi_reduce_min_abs");
      std::string multi_reduce_op_name =
          "System-ClipGradient-GlobalNorm-MultiReduceXimumAbs-" + NewUniqueId();
      auto multi_reduce_op_builder = user_op::UserOpConfWrapperBuilder(multi_reduce_op_name)
                                         .Op(multi_reduce_op_type_name)
                                         .Output("y")
                                         .ScopeSymbolId(scope_symbol_id);
      for (const auto& lbi : lbis) {
        multi_reduce_op_builder.Input("x", GenLogicalBlobName(lbi2diff_lbi.at(lbi)));
      }
      auto multi_reduce_op = multi_reduce_op_builder.Build();
      job_builder->AddOps(parallel_desc.parallel_conf(), {multi_reduce_op.op_conf()});
      if (has_split) {
        std::string group_reduce_op_type_name = max_or_min ? "reduce_max" : "reduce_min";
        std::string group_reduce_op_name =
            "System-ClipGradient-GlobalNorm-GroupReduceXimum-" + NewUniqueId();
        auto group_reduce_op = user_op::UserOpConfWrapperBuilder(group_reduce_op_name)
                                   .Op(group_reduce_op_type_name)
                                   .Input("input_tensor", multi_reduce_op.output("y", 0))
                                   .Output("output_tensor")
                                   .Attr("axis", std::vector<int32_t>{0})
                                   .Attr("keepdims", false)
                                   .ScopeSymbolId(scope_symbol_id)
                                   .Build();
        job_builder->AddOps(parallel_desc.parallel_conf(), {group_reduce_op.op_conf()});
        group_reduce_lbns.push_back(group_reduce_op.output("output_tensor", 0));
      } else {
        group_reduce_lbns.push_back(multi_reduce_op.output("y", 0));
      }
    } else {
      UNIMPLEMENTED();
    }
  };
  ForEachAggregatedParamGroup(op_graph, lbi2diff_lbi, GroupReduce);
  CHECK_GT(group_reduce_lbns.size(), 0);

  *out_parallel_conf = all_same_parallel_desc ? any_parallel_desc.parallel_conf()
                                              : GenParallelConfOfCpuZeroOnMaster();
  out_parallel_conf->mutable_hierarchy()->clear_dim();
  if (group_reduce_lbns.size() == 1) {
    return group_reduce_lbns[0];
  } else {
    // stack all group max and go on max
    const int64_t scope_symbol_id =
        MakeScopeSymbolId(job_builder->job().job_conf(), *out_parallel_conf);
    auto stack_op_builder =
        user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-GlobalStack-"
                                          + NewUniqueId())
            .Op("stack")
            .Output("out")
            .Attr("axis", int64_t(0))
            .Attr("max_dim_size", static_cast<int64_t>(group_reduce_lbns.size()))
            .ScopeSymbolId(scope_symbol_id);
    for (const auto& lbn : group_reduce_lbns) { stack_op_builder.Input("in", lbn); }
    auto stack_op = stack_op_builder.Build();
    job_builder->AddOps(*out_parallel_conf, {stack_op.op_conf()});

    std::string reduce_op_type_name = max_or_min ? "reduce_max" : "reduce_min";
    std::string reduce_op_name =
        "System-ClipGradient-GlobalNorm-GlobalReduceXimum-" + NewUniqueId();
    auto reduce_op = user_op::UserOpConfWrapperBuilder(reduce_op_name)
                         .Op(reduce_op_type_name)
                         .Input("input_tensor", stack_op.output("out", 0))
                         .Output("output_tensor")
                         .Attr("axis", std::vector<int32_t>{0})
                         .Attr("keepdims", false)
                         .ScopeSymbolId(scope_symbol_id)
                         .Build();
    job_builder->AddOps(*out_parallel_conf, {reduce_op.op_conf()});
    return reduce_op.output("output_tensor", 0);
  }
}

std::string GlobalNorm(const OpGraph& op_graph, JobBuilder* job_builder,
                       const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi, float p,
                       ParallelConf* out_parallel_conf) {
  bool all_same_parallel_desc = true;
  const ParallelDesc& any_parallel_desc =
      op_graph.OpNode4OpName(lbi2diff_lbi.begin()->first.op_name())->parallel_desc();
  bool all_broadcast = true;
  std::vector<std::string> group_lbns;
  std::vector<ParallelConf> group_parallel_confs;
  group_lbns.reserve(lbi2diff_lbi.size());
  group_parallel_confs.reserve(lbi2diff_lbi.size());

  auto GroupNorm = [&](const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                       const std::vector<LogicalBlobId>& lbis) {
    if (!parallel_desc.EqualsIgnoringHierarchy(any_parallel_desc)) {
      all_same_parallel_desc = false;
    }
    int64_t scope_symbol_id =
        MakeScopeSymbolId(job_builder->job().job_conf(), parallel_desc.parallel_conf());
    if (!IsBroadcast(nd_sbp, parallel_desc)) { all_broadcast = false; }
    group_parallel_confs.emplace_back(parallel_desc.parallel_conf());

    if (job_builder->job().job_conf().enable_gradients_stats_aggregation()) {
      auto multi_reduce_sum_op_builder =
          user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-MultiReduceSumPowAbs-"
                                            + NewUniqueId())
              .Op("multi_reduce_sum_pow_abs")
              .Attr("p", p)
              .Output("y")
              .ScopeSymbolId(scope_symbol_id);
      for (const auto& lbi : lbis) {
        multi_reduce_sum_op_builder.Input("x", GenLogicalBlobName(lbi2diff_lbi.at(lbi)));
      }
      const auto multi_reduce_sum_op = multi_reduce_sum_op_builder.Build();
      job_builder->AddOps(parallel_desc.parallel_conf(), {multi_reduce_sum_op.op_conf()});
      group_lbns.emplace_back(multi_reduce_sum_op.output("y", 0));
    } else {
      std::vector<std::string> lbns_to_add;
      lbns_to_add.reserve(lbis.size());
      for (const auto& lbi : lbis) {
        const LogicalBlobId& diff_lbi = lbi2diff_lbi.at(lbi);
        const auto square_sum_op =
            user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-ReduceSumPowAbs-"
                                              + NewUniqueId())
                .Op("multi_reduce_sum_pow_abs")
                .Input("x", GenLogicalBlobName(diff_lbi))
                .Attr("p", p)
                .Output("y")
                .ScopeSymbolId(scope_symbol_id)
                .Build();
        job_builder->AddOps(parallel_desc.parallel_conf(), {square_sum_op.op_conf()});
        lbns_to_add.emplace_back(square_sum_op.output("y", 0));
      }
      group_lbns.emplace_back(AddLbns(job_builder, lbns_to_add, parallel_desc.parallel_conf(),
                                      scope_symbol_id, "System-ClipGradient-GlobalNorm-Add-"));
    }
  };
  ForEachAggregatedParamGroup(op_graph, lbi2diff_lbi, GroupNorm);

  // sum in group
  *out_parallel_conf = all_same_parallel_desc ? any_parallel_desc.parallel_conf()
                                              : GenParallelConfOfCpuZeroOnMaster();
  const int64_t scope_symbol_id =
      MakeScopeSymbolId(job_builder->job().job_conf(), *out_parallel_conf);
  std::vector<std::string> sum_group_lbns;
  if (all_broadcast) {
    sum_group_lbns = std::move(group_lbns);
  } else {
    sum_group_lbns.reserve(group_lbns.size());
    for (size_t i = 0; i < group_lbns.size(); ++i) {
      std::string lbn;
      if (all_same_parallel_desc) {
        // reduce many times P->B (allreduce) to 1 times
        lbn = AddParallelCast(job_builder, group_lbns.at(i), "P", group_parallel_confs.at(i),
                              "System-ClipGradient-ParallelCast-");
      } else {
        // sum will run on cpu 0, we need do P->B first,
        // because when execution is on single device, only B is accepted
        lbn = AddParallelCast(job_builder, group_lbns.at(i), "B", group_parallel_confs.at(i),
                              "System-ClipGradient-ParallelCast-");
      }
      sum_group_lbns.push_back(std::move(lbn));
    }
    out_parallel_conf->mutable_hierarchy()->clear_dim();
  }
  auto global_reduce_sum_lbn = AddLbns(job_builder, sum_group_lbns, *out_parallel_conf,
                                       scope_symbol_id, "System-ClipGradient-GlobalNorm-Add-");

  auto global_pow_op =
      user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-GlobalPow-" + NewUniqueId())
          .Op("scalar_pow")
          .Input("in", global_reduce_sum_lbn)
          .Attr("float_operand", 1.0 / p)
          .Attr("has_float_operand", true)
          .Output("out")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(*out_parallel_conf, {global_pow_op.op_conf()});

  return global_pow_op.output("out", 0);
}

void ClipGradientByGlobalNorm(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                              HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi,
                              const ClipByGlobalNormConf& conf) {
  if (lbi2diff_lbi->empty()) { return; }
  ParallelConf parallel_conf;
  std::string total_norm_lbn;
  CHECK(conf.has_norm_type());
  double norm_type = conf.norm_type();
  if (std::isinf(norm_type) && norm_type > 0) {
    total_norm_lbn = GlobalAbsMaxMin(op_graph, job_builder, *lbi2diff_lbi, true, &parallel_conf);
  } else if (std::isinf(norm_type) && norm_type < 0) {
    total_norm_lbn = GlobalAbsMaxMin(op_graph, job_builder, *lbi2diff_lbi, false, &parallel_conf);
  } else {
    total_norm_lbn = GlobalNorm(op_graph, job_builder, *lbi2diff_lbi, norm_type, &parallel_conf);
  }

  int64_t scope_symbol_id = MakeScopeSymbolId(job_builder->job().job_conf(), parallel_conf);

  auto add_eps_ops =
      user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-AddEps-" + NewUniqueId())
          .Op("scalar_add")
          .Input("in", total_norm_lbn)
          .Attr("float_operand", 1e-6)
          .Attr("has_float_operand", true)
          .Output("out")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(parallel_conf, {add_eps_ops.op_conf()});

  auto inv_op =
      user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-Inv-" + NewUniqueId())
          .Op("reciprocal_no_nan")
          .Input("x", add_eps_ops.output("out", 0))
          .Output("y")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(parallel_conf, {inv_op.op_conf()});

  auto coeff_op =
      user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-Coeff-" + NewUniqueId())
          .Op("scalar_mul")
          .Input("in", inv_op.output("y", 0))
          .Attr("float_operand", static_cast<double>(conf.max_norm()))
          .Attr("has_float_operand", true)
          .Output("out")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(parallel_conf, {coeff_op.op_conf()});

  auto clamp_coeff_op =
      user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-Clamp-" + NewUniqueId())
          .Op("clip_by_scalar_max")
          .Input("x", coeff_op.output("out", 0))
          .Attr("floating_max", 1.0)
          .Output("y")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(parallel_conf, {clamp_coeff_op.op_conf()});

  const std::string& coeff_lbn = clamp_coeff_op.output("y", 0);
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    auto mul_op_name = "System-ClipGradient-GlobalNorm-ScalarMul-" + NewUniqueId();
    auto scalar_mul_op = user_op::UserOpConfWrapperBuilder(mul_op_name)
                             .Op("scalar_mul_by_tensor")
                             .Input("x", GenLogicalBlobName(diff_lbi))
                             .Input("scalar", coeff_lbn)
                             .Output("y")
                             .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
                             .Build();
    job_builder->AddOps(op_graph.OpNode4OpName(lbi.op_name())->parallel_desc().parallel_conf(),
                        {scalar_mul_op.op_conf()});
    diff_lbi = GenLogicalBlobId(scalar_mul_op.output("y", 0));
  }

  if (!CHECK_JUST(ctx->HasState<ClipByGlobalNormJobPassState>("clip_by_global_norm_state"))) {
    CHECK_JUST(ctx->ResetState("clip_by_global_norm_state",
                               std::make_unique<ClipByGlobalNormJobPassState>()));
  }
  auto state =
      CHECK_JUST(ctx->MutableState<ClipByGlobalNormJobPassState>("clip_by_global_norm_state"));
  const std::shared_ptr<ClipByGlobalNormJobPassState::TotalNormState>& total_norm_state =
      std::make_shared<ClipByGlobalNormJobPassState::TotalNormState>(
          total_norm_lbn, coeff_lbn, parallel_conf, scope_symbol_id);
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    const std::string& variable_op_name = lbi.op_name();
    state->AddTotalNormState(variable_op_name, total_norm_state);
  }
}

}  // namespace

Maybe<void> MakeGetterLossOpNode4OpName(
    const OpGraph& op_graph, std::function<OpNode*(const std::string&)>* LossOpNode4OpName) {
  std::list<OpNode*> loss_nodes;
  JUST(GetLossOpNodes(op_graph, &loss_nodes));
  auto loss_op_name2op_node = std::make_shared<HashMap<std::string, OpNode*>>();
  for (OpNode* op_node : loss_nodes) {
    CHECK(loss_op_name2op_node->emplace(op_node->op().op_name(), op_node).second);
  }
  *LossOpNode4OpName = [loss_op_name2op_node](const std::string& op_name) -> OpNode* {
    return loss_op_name2op_node->at(op_name);
  };
  return Maybe<void>::Ok();
}

Maybe<void> MakePredicatorNeedBackwardOp(const OpGraph& op_graph,
                                         std::function<bool(OpNode*)>* NeedBackwardOp) {
  auto var_op_nodes_and_descendants = std::make_shared<HashSet<OpNode*>>();
  GetVariableOpNodesAndDescendants(op_graph, var_op_nodes_and_descendants.get());
  auto loss_op_nodes_and_ascendants = std::make_shared<HashSet<OpNode*>>();
  JUST(GetLossOpNodesAndAscendants(op_graph, loss_op_nodes_and_ascendants.get()));
  *NeedBackwardOp = [var_op_nodes_and_descendants, loss_op_nodes_and_ascendants](OpNode* op_node) {
    if (var_op_nodes_and_descendants->find(op_node) == var_op_nodes_and_descendants->end()) {
      return false;
    }
    if (loss_op_nodes_and_ascendants->find(op_node) == loss_op_nodes_and_ascendants->end()) {
      return false;
    }
    for (const auto& ibn : op_node->op().input_bns()) {
      if (op_node->op().InputBlobModifier4Ibn(ibn).requires_grad()) { return true; }
    }
    for (const auto& obn : op_node->op().output_bns()) {
      if (op_node->op().OutputBlobModifier4Obn(obn).requires_grad()) { return true; }
    }
    return false;
  };
  return Maybe<void>::Ok();
}

void GetVariableOpNodesAndDescendants(const OpGraph& op_graph, HashSet<OpNode*>* op_nodes) {
  std::list<OpNode*> starts;
  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (op_conf.has_variable_conf()) { starts.emplace_back(op_node); }
    if (op_conf.has_user_conf()
        && op_conf.user_conf().op_type_name() == "embedding_lookup_placeholder") {
      starts.push_back(op_node);
    }
  });
  auto ForEachNextNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    for (OpEdge* edge : op_node->out_edges()) {
      if (AnyLbiWithDiffLbi(edge)) { Handler(edge->dst_node()); }
    }
  };
  op_graph.BfsForEachNode(starts, ForEachNextNode,
                          [&](OpNode* op_node) { op_nodes->emplace(op_node); });
}

Maybe<void> GenerateBackwardOpConfWrapperStruct::Call(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) const {
  if (naive_func_) {
    (*naive_func_)(op, op_confs, DiffLbi4BnInOp);
  } else if (maybe_func_) {
    JUST((*maybe_func_)(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp));
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "\nNo gradient function found\n"
                                << PbMessage2TxtString(op.op_conf());
  }
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConfIf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  std::unique_ptr<GenerateBackwardOpConfWrapperStruct> obj;
  const auto& op_type_case = op.op_conf().op_type_case();
  if (!IsClassRegistered<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case)) {
    return Error::GradientFunctionNotFoundError() << PbMessage2TxtString(op.op_conf());
  }
  obj.reset(NewObj<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case));
  return obj->Call(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
}

Maybe<void> AutoGrad(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                     HashMap<LogicalBlobId, LogicalBlobId>* out_lbi2out_diff_lbi,
                     OpBlobArgPairs* identical_sbp_oba_pairs) {
  std::function<bool(OpNode*)> NeedBackwardOp;
  JUST(MakePredicatorNeedBackwardOp(op_graph, &NeedBackwardOp));
  std::list<OpNode*> loss_nodes;
  JUST(GetLossOpNodes(op_graph, &loss_nodes));
  CheckNotReachableAmongOpNodes(op_graph, loss_nodes);
  for (OpNode* loss_node : loss_nodes) {
    CHECK(NeedBackwardOp(loss_node)) << loss_node->op().op_name();
  }

  // generate ones lbi as loss's diff
  HashMap<OpBlobArg, LogicalBlobId> out_oba2out_diff_lbi;
  InitOutOba2OutDiffLbi(ctx, op_graph, loss_nodes, &out_oba2out_diff_lbi, job_builder);

  // generate backward ops
  auto ForEachInNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    op_node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (NeedBackwardOp(in_node)) { Handler(in_node); }
    });
  };
  auto ForEachOutNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    op_node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (NeedBackwardOp(out_node)) { Handler(out_node); }
    });
  };
  auto HasDiff4LbiOpName = MakePredicatorHasDiff4LbiOpName(op_graph, NeedBackwardOp);
  HashMap<OpBlobArg, LogicalBlobId> in_oba2in_diff_lbi;
  HashMap<OpBlobArg, LogicalBlobId> out_oba2clone_bw_add_out_lbi;
  std::list<OpNode*> topo_nodes;
  op_graph.TopoForEachNode(loss_nodes, ForEachOutNode, ForEachInNode,
                           [&](OpNode* op_node) { topo_nodes.emplace_back(op_node); });
  for (OpNode* op_node : topo_nodes) {
    const auto& op_name = op_node->op().op_name();
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
      const auto& input_bns = op_node->op().input_bns();
      const auto& output_bns = op_node->op().output_bns();
      if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
        if (HasDiff4LbiOpName(op_node->op().BnInOp2Lbi(bn), op_name) == false) { return nullptr; }
        if (op_node->op().InputBlobModifier4Ibn(bn).requires_grad() == false) { return nullptr; }
        return &in_oba2in_diff_lbi[GenOpBlobArg(op_name, bn)];
      } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
        if (op_node->op().OutputBlobModifier4Obn(bn).requires_grad() == false) { return nullptr; }
        const auto& out_diff_lbi_it = out_oba2out_diff_lbi.find(GenOpBlobArg(op_name, bn));
        if (out_diff_lbi_it == out_oba2out_diff_lbi.end()) { return nullptr; }
        return &out_diff_lbi_it->second;
      } else {
        LOG(FATAL) << "diff lbi for bn in op not found, bn: " << op_name << "/" << bn;
      }
      return nullptr;
    };
    auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return op_graph.GetLogicalBlobDesc(op_node->op().BnInOp2Lbi(bn));
    };
    JUST(GenerateCloneGradOpIfNeed(*op_node, job_builder, in_oba2in_diff_lbi, &out_oba2out_diff_lbi,
                                   &out_oba2clone_bw_add_out_lbi));
    std::vector<OperatorConf> ops;
    JUST(GenerateBackwardOpConfIf(op_node->op(), &ops, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp));
    int64_t scope_symbol_id = op_node->op().op_conf().scope_symbol_id();
    for (auto& op_conf : ops) { op_conf.set_scope_symbol_id(scope_symbol_id); }
    if (op_node->op().op_conf().has_user_conf()
        && op_node->op().op_conf().user_conf().op_type_name() == "hierarchical_parallel_cast") {
      const auto& producer_node = op_node->ProducerOpNode4Lbi(op_node->op().BnInOp2Lbi("in_0"));
      job_builder->AddOps(producer_node.parallel_desc().parallel_conf(), ops);
    } else {
      job_builder->AddOps(op_node->parallel_desc().parallel_conf(), ops);
    }
  }
  CalcOutLbi2OutDiffLbi(op_graph, out_oba2out_diff_lbi, out_lbi2out_diff_lbi);
  return Maybe<void>::Ok();
}

Maybe<void> ScaleModelDiffByLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder,
                                            HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  std::function<OpNode*(const std::string&)> LossOpNode4OpName;
  JUST(MakeGetterLossOpNode4OpName(op_graph, &LossOpNode4OpName));
  const auto& train_conf = GetTrainConf();
  HashMap<LogicalBlobId, OpNode*> loss_lbi2op_node;
  for (const auto& loss_lbn : train_conf.loss_lbn()) {
    const auto& lbi = GenLogicalBlobId(loss_lbn);
    CHECK(loss_lbi2op_node.emplace(lbi, LossOpNode4OpName(lbi.op_name())).second);
  }
  const Shape src_time_shape({1, 1});
  const int64_t source_time_shape_elem_cnt = src_time_shape.elem_cnt();
  bool all_loss_time_shape_eq_src = true;
  for (const auto& pair : loss_lbi2op_node) {
    const int64_t time_shape_elem_cnt = JUST(pair.second->op().GetOpTimeShape())->elem_cnt();
    if (time_shape_elem_cnt != source_time_shape_elem_cnt) {
      CHECK_EQ(time_shape_elem_cnt % source_time_shape_elem_cnt, 0);
      all_loss_time_shape_eq_src = false;
    }
  }
  if (all_loss_time_shape_eq_src) {
    const BlobDesc* blob_desc = nullptr;
    for (const auto& pair : loss_lbi2op_node) {
      const BlobDesc* cur_blob_desc = &pair.second->LogicalBlobDesc4Lbi(pair.first);
      if (blob_desc != nullptr) { CHECK(*blob_desc == *cur_blob_desc); }
      blob_desc = cur_blob_desc;
    }
    if (blob_desc->is_dynamic()) {
      ScaleModelDiffByDynamicLossInstanceNum(op_graph, job_builder, lbi2diff_lbi, loss_lbi2op_node);
    } else {
      ScaleModelDiffByConstantLossInstanceNum(op_graph, job_builder, lbi2diff_lbi,
                                              blob_desc->shape().elem_cnt());
    }
  } else {
    std::unique_ptr<BlobDesc> blob_desc;
    for (const auto& pair : loss_lbi2op_node) {
      const BlobDesc* cur_blob_desc = &pair.second->LogicalBlobDesc4Lbi(pair.first);
      // TODO: support dynamic
      CHECK(!cur_blob_desc->is_dynamic());
      const DataType loss_data_type = cur_blob_desc->data_type();
      const int64_t time_shape_elem_cnt = JUST(pair.second->op().GetOpTimeShape())->elem_cnt();
      // TODO: consider sbp
      const int64_t loss_elem_cnt =
          cur_blob_desc->shape().elem_cnt() * time_shape_elem_cnt / source_time_shape_elem_cnt;
      if (blob_desc) {
        CHECK_EQ(blob_desc->data_type(), loss_data_type);
        CHECK_EQ(blob_desc->shape().elem_cnt(), loss_elem_cnt);
      } else {
        blob_desc.reset(new BlobDesc(Shape({loss_elem_cnt}), loss_data_type));
      }
    }
    ScaleModelDiffByConstantLossInstanceNum(op_graph, job_builder, lbi2diff_lbi,
                                            blob_desc->shape().elem_cnt());
  }
  return Maybe<void>::Ok();
}

void ScaleModelDiffByLossScale(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                               HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  auto ProducerOpNode4Lbi = [&](const LogicalBlobId& lbi) {
    return op_graph.OpNode4OpName(lbi.op_name());
  };
  auto ProducerOpNode4Lbn = [&](const std::string& lbn) {
    return ProducerOpNode4Lbi(GenLogicalBlobId(lbn));
  };
  const TrainConf& train_conf = ctx->job_desc().job_conf().train_conf();
  if (train_conf.has_dynamic_loss_scale_policy()) {
    const auto& dynamic_loss_scale_state =
        CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    HashMap<DataType, std::string> data_type2loss_scale_lbn;
    const auto LossScale4DataType = [&](DataType data_type) -> std::string {
      auto it = data_type2loss_scale_lbn.find(data_type);
      if (it == data_type2loss_scale_lbn.end()) {
        const std::string& loss_scale_val_lbn = dynamic_loss_scale_state.loss_scale_val_lbn();
        const int64_t scope_symbol_id =
            ScopeSymbolId4Lbi(op_graph, GenLogicalBlobId(loss_scale_val_lbn));
        const ParallelConf& parallel_conf =
            ProducerOpNode4Lbn(loss_scale_val_lbn)->parallel_desc().parallel_conf();
        std::string loss_scale_lbn_with_data_type;
        if (data_type == DataType::kFloat) {
          loss_scale_lbn_with_data_type = loss_scale_val_lbn;
        } else {
          auto cast_op =
              user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-Cast-" + NewUniqueId())
                  .Op("cast")
                  .Input("in", loss_scale_val_lbn)
                  .Output("out")
                  .Attr<DataType>("dtype", data_type)
                  .ScopeSymbolId(scope_symbol_id)
                  .Build();
          loss_scale_lbn_with_data_type = cast_op.output("out", 0);
          job_builder->AddOps(parallel_conf, {cast_op.op_conf()});
        }
        auto inv_scale_op =
            user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-Reciprocal-" + NewUniqueId())
                .Op("reciprocal")
                .Input("x", loss_scale_lbn_with_data_type)
                .Output("y")
                .ScopeSymbolId(scope_symbol_id)
                .Build();
        job_builder->AddOps(parallel_conf, {inv_scale_op.op_conf()});
        std::string lbn = inv_scale_op.output("y", 0);
        data_type2loss_scale_lbn[data_type] = lbn;
        return lbn;
      } else {
        return it->second;
      }
    };
    for (auto& pair : *lbi2diff_lbi) {
      const LogicalBlobId& lbi = pair.first;
      LogicalBlobId& diff_lbi = pair.second;
      auto scalar_mul_op =
          user_op::UserOpConfWrapperBuilder("System-ModelDiffScale-ScalarMul-" + NewUniqueId())
              .Op("scalar_mul_by_tensor")
              .Input("x", GenLogicalBlobName(diff_lbi))
              .Input("scalar", LossScale4DataType(op_graph.GetLogicalBlobDesc(lbi).data_type()))
              .Output("y")
              .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
              .Build();
      job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {scalar_mul_op.op_conf()});
      diff_lbi = GenLogicalBlobId(scalar_mul_op.output("y", 0));
    }
  } else if (train_conf.has_loss_scale_factor()) {
    const float loss_scale_factor = train_conf.loss_scale_factor();
    if (loss_scale_factor == 1) { return; }
    const float down_scale_factor = 1.0f / loss_scale_factor;
    for (auto& pair : *lbi2diff_lbi) {
      const LogicalBlobId& lbi = pair.first;
      LogicalBlobId& diff_lbi = pair.second;
      auto scalar_mul_op =
          user_op::UserOpConfWrapperBuilder("System-ModelDiffScale-ScalarMul-" + NewUniqueId())
              .Op("scalar_mul")
              .Input("in", GenLogicalBlobName(diff_lbi))
              .Output("out")
              .Attr<bool>("has_float_operand", true)
              .Attr<double>("float_operand", down_scale_factor)
              .Attr<bool>("has_int_operand", false)
              .Attr<int64_t>("int_operand", 0)
              .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
              .Build();
      job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {scalar_mul_op.op_conf()});
      diff_lbi = GenLogicalBlobId(scalar_mul_op.output("out", 0));
    }
  } else {
    return;
  }
}

void RegularizeGradient(const OpGraph& op_graph, JobBuilder* job_builder,
                        HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    int64_t scope_symbol_id = model_op_node->op().op_conf().scope_symbol_id();
    CHECK(model_op_node->op().op_conf().has_variable_conf());
    const VariableOpConf& variable_conf = model_op_node->op().op_conf().variable_conf();
    if (!variable_conf.has_regularizer()) { continue; }
    const RegularizerConf& regularizer_conf = variable_conf.regularizer();
    if (regularizer_conf.has_l1_l2_conf()) {
      user_op::UserOpConfWrapper regularize_gradient_op =
          user_op::UserOpConfWrapperBuilder("System-RegularizeGradient-L1L2-" + NewUniqueId())
              .Op("l1_l2_regularize_gradient")
              .Input("model", GenLogicalBlobName(lbi))
              .Input("model_diff", GenLogicalBlobName(diff_lbi))
              .Output("out")
              .Attr<float>("l1", regularizer_conf.l1_l2_conf().l1())
              .Attr<float>("l2", regularizer_conf.l1_l2_conf().l2())
              .ScopeSymbolId(scope_symbol_id)
              .Build();
      job_builder->AddOps(model_op_node->parallel_desc().parallel_conf(),
                          {regularize_gradient_op.op_conf()});
      diff_lbi = GenLogicalBlobId(regularize_gradient_op.output("out", 0));
    } else {
      UNIMPLEMENTED();
    }
  }
}

void ClipGradient(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                  HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi, const ClipConf& clip_conf) {
  if (clip_conf.has_clip_by_global_norm()) {
    ClipGradientByGlobalNorm(ctx, op_graph, job_builder, lbi2diff_lbi,
                             clip_conf.clip_by_global_norm());
  } else {
    UNIMPLEMENTED();
  }
}

void AddDiffParallelCast(const OpGraph& op_graph, JobBuilder* job_builder,
                         HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    if (model_op_node->parallel_desc().parallel_num() <= 1) { continue; }
    const int64_t scope_symbol_id = model_op_node->op().op_conf().scope_symbol_id();
    std::vector<std::string> nd_sbp;
    const std::string& variable_sole_obn = model_op_node->op().SoleObn();
    nd_sbp.reserve(model_op_node->NdSbp4BnInOp(variable_sole_obn).sbp_parallel().size());
    for (const auto& sbp_parallel : model_op_node->NdSbp4BnInOp(variable_sole_obn).sbp_parallel()) {
      nd_sbp.emplace_back(SbpParallelToString(sbp_parallel));
    }
    auto parallel_cast_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-ParallelCast-" + NewUniqueId())
            .Op("hierarchical_parallel_cast")
            .Input("in", GenLogicalBlobName(diff_lbi))
            .Output("out")
            .Attr<std::vector<std::string>>("nd_sbp", nd_sbp)
            .Attr<std::string>("grad_mode", "auto")
            .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>())
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder->AddOps(model_op_node->parallel_desc().parallel_conf(),
                        {parallel_cast_op.op_conf()});
    diff_lbi = GenLogicalBlobId(parallel_cast_op.output("out", 0));
  }
}

void AddDiffStaticShapeCast(const OpGraph& op_graph, JobBuilder* job_builder,
                            HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    int64_t scope_symbol_id = model_op_node->op().op_conf().scope_symbol_id();
    const auto cast_to_static_shape_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-StaticShapeCast-" + NewUniqueId())
            .Op("cast_to_static_shape")
            .Input("input", GenLogicalBlobName(diff_lbi))
            .Output("output")
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder->AddOps(model_op_node->parallel_desc().parallel_conf(),
                        {cast_to_static_shape_op.op_conf()});
    diff_lbi = GenLogicalBlobId(cast_to_static_shape_op.output("output", 0));
  }
}

Maybe<void> CountNotFiniteIfNeeded(JobPassCtx* ctx, const OpGraph& op_graph,
                                   JobBuilder* job_builder,
                                   const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi) {
  if (lbi2diff_lbi.empty()) { return Maybe<void>::Ok(); }
  if (!ctx->job_desc().job_conf().train_conf().has_dynamic_loss_scale_policy()) {
    return Maybe<void>::Ok();
  }
  bool all_same_parallel_desc = true;
  const ParallelDesc& any_parallel_desc =
      op_graph.OpNode4OpName(lbi2diff_lbi.begin()->first.op_name())->parallel_desc();
  std::vector<std::string> partial_count_not_finite_lbns;
  std::vector<bool> is_broadcast_nd_sbp;
  std::vector<ParallelConf> param_group_parallel_confs;
  ForEachAggregatedParamGroup(
      op_graph, lbi2diff_lbi,
      [&](const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
          const std::vector<LogicalBlobId>& lbis) {
        if (!parallel_desc.EqualsIgnoringHierarchy(any_parallel_desc)) {
          all_same_parallel_desc = false;
        }
        const int64_t scope_symbol_id =
            MakeScopeSymbolId(job_builder->job().job_conf(), parallel_desc.parallel_conf());
        is_broadcast_nd_sbp.emplace_back(IsBroadcast(nd_sbp, parallel_desc));
        param_group_parallel_confs.emplace_back(parallel_desc.parallel_conf());
        if (job_builder->job().job_conf().enable_gradients_stats_aggregation()) {
          auto multi_count_not_finite_op_builder =
              user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-MultiCountNotFinite-"
                                                + NewUniqueId())
                  .Op("multi_count_not_finite")
                  .Output("y")
                  .ScopeSymbolId(scope_symbol_id);
          for (const auto& lbi : lbis) {
            multi_count_not_finite_op_builder.Input("x", GenLogicalBlobName(lbi2diff_lbi.at(lbi)));
          }
          const auto multi_count_not_finite_op = multi_count_not_finite_op_builder.Build();
          job_builder->AddOps(parallel_desc.parallel_conf(), {multi_count_not_finite_op.op_conf()});
          partial_count_not_finite_lbns.emplace_back(multi_count_not_finite_op.output("y", 0));
        } else {
          std::vector<std::string> lbns_to_add;
          for (const auto& lbi : lbis) {
            const auto count_not_finite_op =
                user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-CountNotFinite-"
                                                  + NewUniqueId())
                    .Op("count_not_finite")
                    .Input("x", GenLogicalBlobName(lbi2diff_lbi.at(lbi)))
                    .Output("y")
                    .ScopeSymbolId(scope_symbol_id)
                    .Build();
            job_builder->AddOps(parallel_desc.parallel_conf(), {count_not_finite_op.op_conf()});
            lbns_to_add.emplace_back(count_not_finite_op.output("y", 0));
          }
          partial_count_not_finite_lbns.emplace_back(
              AddLbns(job_builder, lbns_to_add, parallel_desc.parallel_conf(), scope_symbol_id,
                      "System-DynamicLossScale-CountNotFinite-Add-"));
        }
      });

  const bool all_group_broadcast =
      std::all_of(is_broadcast_nd_sbp.begin(), is_broadcast_nd_sbp.end(), [](bool i) { return i; });
  std::vector<std::string> count_not_finite_lbns_for_add;
  ParallelConf count_all_parallel_conf = all_same_parallel_desc
                                             ? any_parallel_desc.parallel_conf()
                                             : GenParallelConfOfCpuZeroOnMaster();
  if (!all_group_broadcast) {
    for (int64_t i = 0; i < partial_count_not_finite_lbns.size(); ++i) {
      count_not_finite_lbns_for_add.emplace_back(AddParallelCast(
          job_builder, JUST(VectorAt(partial_count_not_finite_lbns, i)), "P",
          JUST(VectorAt(param_group_parallel_confs, i)), "System-DynamicLossScale-ParallelCast-"));
    }
    count_all_parallel_conf.mutable_hierarchy()->clear_dim();
  } else {
    count_not_finite_lbns_for_add = std::move(partial_count_not_finite_lbns);
  }
  const int64_t scope_symbol_id =
      MakeScopeSymbolId(job_builder->job().job_conf(), count_all_parallel_conf);
  std::string count_all_lbn =
      AddLbns(job_builder, count_not_finite_lbns_for_add, count_all_parallel_conf, scope_symbol_id,
              "System-DynamicLossScale-CountNotFinite-Add-");
  if (!all_group_broadcast) {
    std::vector<std::string> cast_nd_sbp;
    cast_nd_sbp.emplace_back("B");
    auto parallel_cast_op =
        user_op::UserOpConfWrapperBuilder(
            "System-DynamicLossScale-CountNotFinite-After-Add-ParallelCast-" + NewUniqueId())
            .Op("hierarchical_parallel_cast")
            .Input("in", count_all_lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("nd_sbp", cast_nd_sbp)
            .Attr<std::string>("grad_mode", "auto")
            .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>())
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder->AddOps(count_all_parallel_conf, {parallel_cast_op.op_conf()});
    count_all_lbn = parallel_cast_op.output("out", 0);
  }
  const LogicalBlobId count_not_finite_lbi =
      GenLogicalBlobId(JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"))
                           .count_not_finite_lbn());
  auto count_not_finite_op = user_op::UserOpConfWrapperBuilder(count_not_finite_lbi.op_name())
                                 .Op("identity")
                                 .Input("in", count_all_lbn)
                                 .Output("out")
                                 .ScopeSymbolId(scope_symbol_id)
                                 .Build();
  job_builder->MutOpsOnlyOnce({count_not_finite_op.op_conf()});
  job_builder->MutParallelConfOnlyOnce(count_not_finite_op.op_name(), count_all_parallel_conf);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
