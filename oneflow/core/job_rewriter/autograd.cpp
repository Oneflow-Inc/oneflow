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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/dynamic_loss_scale_job_pass_state.h"

namespace oneflow {

namespace {

const TrainConf& GetTrainConf() { return GlobalJobDesc().job_conf().train_conf(); }

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
      loss_op_nodes->push_back(op_node);
    }
  });
  if (loss_op_nodes->empty()) { return Error::LossBlobNotFoundError("Loss blob not found."); }
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
                           const LogicalBlobId& lbi, std::vector<OperatorConf>* op_confs,
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
  op_confs->push_back(constant_like_op);
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
              .Build();
      OpBlobArg cast_in_op_blob_arg;
      cast_in_op_blob_arg.set_op_name(cast_op.op_name());
      cast_in_op_blob_arg.set_bn_in_op(GenRepeatedBn("in", 0));
      job_builder->MutSbpParallel4Oba(cast_in_op_blob_arg)->mutable_broadcast_parallel();
      OpBlobArg cast_out_op_blob_arg;
      cast_out_op_blob_arg.set_op_name(cast_op.op_name());
      cast_out_op_blob_arg.set_bn_in_op(GenRepeatedBn("out", 0));
      job_builder->MutSbpParallel4Oba(cast_out_op_blob_arg)->mutable_broadcast_parallel();
      op_confs->push_back(cast_op.op_conf());
      loss_scale_val_lbn = cast_op.output("out", 0);
    }
    // TODO(liujuncheng): The time shapes of loss and scale may be different.
    auto scalar_mul_op =
        user_op::UserOpConfWrapperBuilder(lbi.op_name() + "_" + lbi.blob_name() + "_grad_Scale")
            .Op("scalar_mul_by_tensor")
            .Input("x", GenLogicalBlobName(constant_like_op.name(), constant_like_conf->out()))
            .Input("scalar", loss_scale_val_lbn)
            .Output("y")
            .Build();
    op_confs->push_back(scalar_mul_op.op_conf());
    *out_diff_lbi = GenLogicalBlobId(scalar_mul_op.output("y", 0));
  } else {
    out_diff_lbi->set_op_name(constant_like_op.name());
    out_diff_lbi->set_blob_name(constant_like_conf->out());
  }
}

const ParallelConf& ProducerParallelConf4Lbi(const OpGraph& op_graph, const LogicalBlobId& lbi) {
  return op_graph.OpNode4OpName(lbi.op_name())->parallel_desc().parallel_conf();
}

int64_t ScopeSymbolId4Lbi(const OpGraph& op_graph, const LogicalBlobId& lbi) {
  return op_graph.OpNode4OpName(lbi.op_name())->op().op_conf().scope_symbol_id();
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

Maybe<void> TryMirroredCastTotalLossInstanceNum(
    JobBuilder* job_builder, const HashMap<LogicalBlobId, OpNode*>& loss_lbi2loss_node,
    LogicalBlobId* total_loss_instance_num_lbi) {
  auto IsMirrored4Lbi = [](const LogicalBlobId& lbi, OpNode* op_node) -> Maybe<bool> {
    const auto& obn = *JUST(op_node->op().obn4lbi(lbi));
    const auto& opt_mirrored_parallel = *JUST(op_node->op().OptMirroredParallel4BnInOp(obn));
    return opt_mirrored_parallel.has_mirrored_parallel();
  };
  const auto& begin = *loss_lbi2loss_node.begin();
  bool is_mirrored = JUST(IsMirrored4Lbi(begin.first, begin.second));
  for (const auto& pair : loss_lbi2loss_node) {
    bool is_other_mirrored = JUST(IsMirrored4Lbi(pair.first, pair.second));
    CHECK_EQ_OR_RETURN(is_mirrored, is_other_mirrored);
  }
  if (is_mirrored) {
    OperatorConf op_conf;
    op_conf.set_name("System-Cast-Mirrored-TotalLossInstanceNum" + NewUniqueId());
    CastFromMirroredOpConf* cast_from_mirrored = op_conf.mutable_cast_from_mirrored_conf();
    cast_from_mirrored->set_in(GenLogicalBlobName(*total_loss_instance_num_lbi));
    cast_from_mirrored->set_out("out");
    cast_from_mirrored->mutable_sbp_parallel()->mutable_partial_sum_parallel();
    const auto& parallel_conf = job_builder->ParallelConf4Lbi(*total_loss_instance_num_lbi);
    int64_t scope_symbol_id = 0;
    {
      const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf =
          std::make_shared<cfg::JobConfigProto>(job_builder->job().job_conf());
      const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf =
          std::make_shared<cfg::ParallelConf>(parallel_conf);
      scope_symbol_id =
          Global<ForeignCallback>::Get()->MakeScopeSymbol(cfg_job_conf, cfg_parallel_conf, true);
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
    instance_num_op_conf->mutable_include_axis_conf()->add_axis(0);
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
      const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf =
          std::make_shared<cfg::JobConfigProto>(job_builder->job().job_conf());
      const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf =
          std::make_shared<cfg::ParallelConf>(parallel_conf);
      scope_symbol_id =
          Global<ForeignCallback>::Get()->MakeScopeSymbol(cfg_job_conf, cfg_parallel_conf, false);
    }
    op_conf.set_scope_symbol_id(scope_symbol_id);
    job_builder->AddOps(parallel_conf, {op_conf});

    total_loss_instance_num_lbi.set_op_name(op_conf.name());
    total_loss_instance_num_lbi.set_blob_name("out");
  } else {
    UNIMPLEMENTED();
  }
  CHECK_JUST(TryMirroredCastTotalLossInstanceNum(job_builder, loss_lbi2loss_node,
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

void BindFwBwObaPairs(const OpGraph& op_graph, const OpBlobArgPairs& fw_bw_oba_pairs,
                      JobBuilder* job_builder) {
  HashSet<OpBlobArg> split_paralleled_obas;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto TryInserSplitParalleledObas = [&](const std::string& bn) {
      const auto& lbi = op_node->op().BnInOp2Lbi(bn);
      const auto& fw_sbp_parallel = op_node->SbpParallel4Lbi(lbi);
      if (fw_sbp_parallel.has_split_parallel()) {
        split_paralleled_obas.insert(GenOpBlobArg(op_node->op().op_name(), bn));
      }
    };
    for (const auto& ibn : op_node->op().input_bns()) { TryInserSplitParalleledObas(ibn); }
    for (const auto& obn : op_node->op().input_bns()) { TryInserSplitParalleledObas(obn); }
  });
  for (const auto& pair : fw_bw_oba_pairs.pair()) {
    CHECK(split_paralleled_obas.find(pair.first()) == split_paralleled_obas.end());
    if (split_paralleled_obas.find(pair.second()) != split_paralleled_obas.end()) {
      job_builder->BindIdenticalSbpOpBlobArgPair(pair.first(), pair.second());
    }
  }
}

void CalcFwBwObaPairs(const OpGraph& op_graph,
                      const HashMap<OpBlobArg, LogicalBlobId>& in_oba2in_diff_lbi,
                      const HashMap<OpBlobArg, LogicalBlobId>& out_oba2out_diff_lbi,
                      const HashMap<OpBlobArg, LogicalBlobId>& out_oba2clone_bw_add_out_lbi,
                      const JobBuilder& job_builder, OpBlobArgPairs* fw_bw_oba_pairs) {
  HashMap<LogicalBlobId, OpBlobArg> in_diff_lbi2in_oba;
  op_graph.ReverseTopoForEachNode([&](OpNode* op_node) {
    const auto& op = op_node->op();
    for (const auto& ibn : op.input_bns()) {
      const auto& in_diff_lbi_it = in_oba2in_diff_lbi.find(GenOpBlobArg(op.op_name(), ibn));
      if (in_diff_lbi_it == in_oba2in_diff_lbi.end()) { continue; }
      if (in_diff_lbi2in_oba.find(in_diff_lbi_it->second) == in_diff_lbi2in_oba.end()) {
        in_diff_lbi2in_oba[in_diff_lbi_it->second] = in_diff_lbi_it->first;
      }
    }
  });
  HashMap<LogicalBlobId, OpBlobArg> out_diff_lbi2out_oba;
  op_graph.TopoForEachNode([&](OpNode* op_node) {
    const auto& op = op_node->op();
    for (const auto& obn : op.output_bns()) {
      const auto& out_diff_lbi_it = out_oba2out_diff_lbi.find(GenOpBlobArg(op.op_name(), obn));
      if (out_diff_lbi_it == out_oba2out_diff_lbi.end()) { continue; }
      if (out_diff_lbi2out_oba.find(out_diff_lbi_it->second) == out_diff_lbi2out_oba.end()) {
        out_diff_lbi2out_oba[out_diff_lbi_it->second] = out_diff_lbi_it->first;
      }
    }
  });
  HashMap<LogicalBlobId, OpBlobArg> clone_bw_add_out_lbi2out_oba;
  for (const auto& pair : out_oba2clone_bw_add_out_lbi) {
    CHECK(clone_bw_add_out_lbi2out_oba.emplace(pair.second, pair.first).second);
  }
  job_builder.ForEachOperator([&](const Operator& op) {
    for (const auto& ibn : op.input_bns()) {
      const auto& out_oba_it = out_diff_lbi2out_oba.find(op.BnInOp2Lbi(ibn));
      if (out_oba_it == out_diff_lbi2out_oba.end()) { continue; }
      auto* pair = fw_bw_oba_pairs->mutable_pair()->Add();
      *pair->mutable_first() = GenOpBlobArg(op.op_name(), ibn);
      *pair->mutable_second() = out_oba_it->second;
    }
    for (const auto& obn : op.output_bns()) {
      const auto& lbi = op.BnInOp2Lbi(obn);
      {
        const auto& in_oba_it = in_diff_lbi2in_oba.find(lbi);
        if (in_oba_it == in_diff_lbi2in_oba.end()) { continue; }
        auto* pair = fw_bw_oba_pairs->mutable_pair()->Add();
        *pair->mutable_first() = GenOpBlobArg(op.op_name(), obn);
        *pair->mutable_second() = in_oba_it->second;
      }
      {
        const auto& clone_out_oba_it = clone_bw_add_out_lbi2out_oba.find(lbi);
        if (clone_out_oba_it == clone_bw_add_out_lbi2out_oba.end()) { continue; }
        auto* pair = fw_bw_oba_pairs->mutable_pair()->Add();
        *pair->mutable_first() = GenOpBlobArg(op.op_name(), obn);
        *pair->mutable_second() = clone_out_oba_it->second;
      }
    }
  });
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
    std::vector<OperatorConf> ops;
    GenerateOriginDiffLbi(ctx, op_graph, job_builder, loss_lbi, &ops, out_diff_lbi);
    int64_t scope_symbol_id = loss_op_node->op().op_conf().scope_symbol_id();
    for (auto& op : ops) { op.set_scope_symbol_id(scope_symbol_id); }
    job_builder->AddOps(loss_op_node->parallel_desc().parallel_conf(), ops);
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
    const std::function<void(const ParallelDesc& parallel_desc, const SbpParallel& sbp_parallel,
                             const std::vector<LogicalBlobId>& libs)>& Handler) {
  HashMap<LogicalBlobId, const ParallelDesc*> lbi2parallel_desc;
  HashMap<std::pair<ParallelDesc, SbpParallel>, std::vector<LogicalBlobId>> group;
  for (auto& pair : lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    const ParallelDesc& parallel_desc = model_op_node->parallel_desc();
    const SbpParallel& sbp_parallel = model_op_node->SbpParallel4Lbi(lbi);
    group[std::make_pair(parallel_desc, sbp_parallel)].push_back(lbi);
  }
  for (const auto& pair : group) { Handler(pair.first.first, pair.first.second, pair.second); }
}

int64_t MakeScopeSymbolId(const JobConfigProto& job_conf, const ParallelConf& parallel_conf) {
  const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf =
      std::make_shared<cfg::JobConfigProto>(job_conf);
  const std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf =
      std::make_shared<cfg::ParallelConf>(parallel_conf);
  return Global<ForeignCallback>::Get()->MakeScopeSymbol(cfg_job_conf, cfg_parallel_conf, false);
}

std::string AddLbns(JobBuilder* job_builder, const std::vector<std::string>& lbns,
                    const ParallelConf& parallel_conf, int64_t scope_symbol_id,
                    const std::string& op_name_prefix) {
  std::vector<std::string> lbns_to_add(lbns);
  const size_t add_n_op_max_input_num = 8;
  while (lbns_to_add.size() != 1) {
    user_op::UserOpConfWrapperBuilder add_op_builder(op_name_prefix + NewUniqueId());
    add_op_builder.Op("add_n");
    const size_t start = lbns_to_add.size() >= add_n_op_max_input_num
                             ? lbns_to_add.size() - add_n_op_max_input_num
                             : 0;
    for (size_t i = start; i < lbns_to_add.size(); ++i) {
      add_op_builder.Input("in", lbns_to_add.at(i));
    }
    lbns_to_add.resize(start);
    const auto add_op = add_op_builder.Output("out").ScopeSymbolId(scope_symbol_id).Build();
    job_builder->AddOps(parallel_conf, {add_op.op_conf()});
    lbns_to_add.push_back(add_op.output("out", 0));
  }
  return lbns_to_add.front();
}

void ClipGradientByGlobalNorm(const OpGraph& op_graph, JobBuilder* job_builder,
                              HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi,
                              const ClipByGlobalNormConf& conf) {
  if (lbi2diff_lbi->empty()) { return; }
  bool all_same_parallel_desc = true;
  const ParallelDesc& any_parallel_desc =
      op_graph.OpNode4OpName(lbi2diff_lbi->begin()->first.op_name())->parallel_desc();
  std::vector<std::string> partial_square_sum_lbns;
  ForEachAggregatedParamGroup(
      op_graph, *lbi2diff_lbi,
      [&](const ParallelDesc& parallel_desc, const SbpParallel& sbp_parallel,
          const std::vector<LogicalBlobId>& lbis) {
        if (parallel_desc != any_parallel_desc) { all_same_parallel_desc = false; }
        int64_t scope_symbol_id =
            MakeScopeSymbolId(job_builder->job().job_conf(), parallel_desc.parallel_conf());
        if (job_builder->job().job_conf().enable_gradients_stats_aggregation()) {
          auto multi_square_sum_op_builder =
              user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-MultiSquareSum-"
                                                + NewUniqueId())
                  .Op("multi_square_sum")
                  .Output("y")
                  .ScopeSymbolId(scope_symbol_id);
          for (const auto& lbi : lbis) {
            multi_square_sum_op_builder.Input("x", GenLogicalBlobName(lbi2diff_lbi->at(lbi)));
          }
          const auto multi_square_sum_op = multi_square_sum_op_builder.Build();
          job_builder->AddOps(parallel_desc.parallel_conf(), {multi_square_sum_op.op_conf()});
          partial_square_sum_lbns.push_back(multi_square_sum_op.output("y", 0));
        } else {
          std::vector<std::string> lbns_to_add;
          for (const auto& lbi : lbis) {
            const LogicalBlobId& diff_lbi = lbi2diff_lbi->at(lbi);
            const auto square_sum_op =
                user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-SquareSum-"
                                                  + NewUniqueId())
                    .Op("square_sum")
                    .Input("x", GenLogicalBlobName(diff_lbi))
                    .Output("y")
                    .ScopeSymbolId(scope_symbol_id)
                    .Build();
            job_builder->AddOps(parallel_desc.parallel_conf(), {square_sum_op.op_conf()});
            lbns_to_add.push_back(square_sum_op.output("y", 0));
          }
          partial_square_sum_lbns.push_back(AddLbns(job_builder, lbns_to_add,
                                                    parallel_desc.parallel_conf(), scope_symbol_id,
                                                    "System-ClipGradient-GlobalNorm-Add-"));
        }
      });

  const ParallelConf global_norm_parallel_conf = all_same_parallel_desc
                                                     ? any_parallel_desc.parallel_conf()
                                                     : GenParallelConfOfCpuZeroOnMaster();
  const int64_t scope_symbol_id =
      MakeScopeSymbolId(job_builder->job().job_conf(), global_norm_parallel_conf);
  const std::string square_sum_lbn =
      AddLbns(job_builder, partial_square_sum_lbns, global_norm_parallel_conf, scope_symbol_id,
              "System-ClipGradient-GlobalNorm-Add-");
  auto inv_global_norm_op = user_op::UserOpConfWrapperBuilder(
                                "System-ClipGradient-GlobalNorm-InvGlobalNorm-" + NewUniqueId())
                                .Op("rsqrt")
                                .Input("x", square_sum_lbn)
                                .Output("y")
                                .ScopeSymbolId(scope_symbol_id)
                                .Build();
  job_builder->AddOps(global_norm_parallel_conf, {inv_global_norm_op.op_conf()});
  OperatorConf inv_clip_norm_op_conf{};
  inv_clip_norm_op_conf.set_name("System-ClipGradient-GlobalNorm-InvClipNorm-" + NewUniqueId());
  ConstantLikeOpConf* inv_clip_norm_constant_like_conf =
      inv_clip_norm_op_conf.mutable_constant_like_conf();
  inv_clip_norm_constant_like_conf->set_like(inv_global_norm_op.output("y", 0));
  inv_clip_norm_constant_like_conf->set_float_operand(1.0 / conf.clip_norm());
  inv_clip_norm_constant_like_conf->set_out("out");
  inv_clip_norm_op_conf.set_scope_symbol_id(scope_symbol_id);
  job_builder->AddOps(global_norm_parallel_conf, {inv_clip_norm_op_conf});
  auto minimum_op =
      user_op::UserOpConfWrapperBuilder("System-ClipGradient-GlobalNorm-Minimum-" + NewUniqueId())
          .Op("broadcast_minimum")
          .Input("x", inv_global_norm_op.output("y", 0))
          .Input("y", GenLogicalBlobName(inv_clip_norm_op_conf.name(),
                                         inv_clip_norm_constant_like_conf->out()))
          .Output("z")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(global_norm_parallel_conf, {minimum_op.op_conf()});
  const std::string gradient_scale_factor_lbn = minimum_op.output("z", 0);
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    auto scalar_mul_op = user_op::UserOpConfWrapperBuilder(
                             "System-ClipGradient-GlobalNorm-ScalarMul-" + NewUniqueId())
                             .Op("scalar_mul_by_tensor")
                             .Input("x", GenLogicalBlobName(diff_lbi))
                             .Input("scalar", gradient_scale_factor_lbn)
                             .Output("y")
                             .ScopeSymbolId(ScopeSymbolId4Lbi(op_graph, lbi))
                             .Build();
    job_builder->AddOps(op_graph.OpNode4OpName(lbi.op_name())->parallel_desc().parallel_conf(),
                        {scalar_mul_op.op_conf()});
    diff_lbi = GenLogicalBlobId(scalar_mul_op.output("y", 0));
  }
}

}  // namespace

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
    if (op_conf.has_variable_conf()) { starts.push_back(op_node); }
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
    return Error::GradientFunctionNotFound() << PbMessage2TxtString(op.op_conf());
  }
  obj.reset(NewObj<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case));
  return obj->Call(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
}

Maybe<void> AutoGrad(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                     HashMap<LogicalBlobId, LogicalBlobId>* out_lbi2out_diff_lbi) {
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
                           [&](OpNode* op_node) { topo_nodes.push_back(op_node); });
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
    GenerateCloneGradOpIfNeed(*op_node, job_builder, in_oba2in_diff_lbi, &out_oba2out_diff_lbi,
                              &out_oba2clone_bw_add_out_lbi);
    std::vector<OperatorConf> ops;
    JUST(GenerateBackwardOpConfIf(op_node->op(), &ops, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp));
    int64_t scope_symbol_id = op_node->op().op_conf().scope_symbol_id();
    for (auto& op_conf : ops) { op_conf.set_scope_symbol_id(scope_symbol_id); }
    job_builder->AddOps(op_node->parallel_desc().parallel_conf(), ops);
  }
  OpBlobArgPairs fw_bw_oba_pairs;
  CalcFwBwObaPairs(op_graph, in_oba2in_diff_lbi, out_oba2out_diff_lbi, out_oba2clone_bw_add_out_lbi,
                   *job_builder, &fw_bw_oba_pairs);
  BindFwBwObaPairs(op_graph, fw_bw_oba_pairs, job_builder);
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
  const Shape src_time_shape(
      {GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()});
  const int64_t source_time_shape_elem_cnt = src_time_shape.elem_cnt();
  bool all_loss_time_shape_eq_src = true;
  for (const auto& pair : loss_lbi2op_node) {
    const Shape* time_shape = pair.second->out_blob_time_shape();
    const int64_t time_shape_elem_cnt = time_shape->elem_cnt();
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
      const int64_t time_shape_elem_cnt = pair.second->out_blob_time_shape()->elem_cnt();
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

void ClipGradient(const OpGraph& op_graph, JobBuilder* job_builder,
                  HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi, const ClipConf& clip_conf) {
  if (clip_conf.has_clip_by_global_norm()) {
    ClipGradientByGlobalNorm(op_graph, job_builder, lbi2diff_lbi, clip_conf.clip_by_global_norm());
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
    int64_t scope_symbol_id = model_op_node->op().op_conf().scope_symbol_id();
    const SbpParallel& model_sbp = model_op_node->SbpParallel4Lbi(lbi);
    auto parallel_cast_op =
        user_op::UserOpConfWrapperBuilder("System-AutoGrad-ParallelCast-" + NewUniqueId())
            .Op("parallel_cast")
            .Input("in", GenLogicalBlobName(diff_lbi))
            .Output("out")
            .Attr("sbp_parallel", SbpParallelToString(model_sbp))
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
  ForEachAggregatedParamGroup(
      op_graph, lbi2diff_lbi,
      [&](const ParallelDesc& parallel_desc, const SbpParallel& sbp_parallel,
          const std::vector<LogicalBlobId>& lbis) {
        if (parallel_desc != any_parallel_desc) { all_same_parallel_desc = false; }
        const int64_t scope_symbol_id =
            MakeScopeSymbolId(job_builder->job().job_conf(), parallel_desc.parallel_conf());
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
          partial_count_not_finite_lbns.push_back(multi_count_not_finite_op.output("y", 0));
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
            lbns_to_add.push_back(count_not_finite_op.output("y", 0));
          }
          partial_count_not_finite_lbns.push_back(
              AddLbns(job_builder, lbns_to_add, parallel_desc.parallel_conf(), scope_symbol_id,
                      "System-DynamicLossScale-CountNotFinite-Add-"));
        }
      });

  const ParallelConf count_all_parallel_conf = all_same_parallel_desc
                                                   ? any_parallel_desc.parallel_conf()
                                                   : GenParallelConfOfCpuZeroOnMaster();
  const int64_t scope_symbol_id =
      MakeScopeSymbolId(job_builder->job().job_conf(), count_all_parallel_conf);
  const std::string count_all_lbn =
      AddLbns(job_builder, partial_count_not_finite_lbns, count_all_parallel_conf, scope_symbol_id,
              "System-DynamicLossScale-CountNotFinite-Add-");
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
