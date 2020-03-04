#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job_completer/clone_grad.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/register/op_blob_arg.pb.h"

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

void GetLossOpNodes(const OpGraph& op_graph, std::list<OpNode*>* loss_op_nodes) {
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
  CHECK_GT(loss_op_nodes->size(), 0);
}

void GetLossOpNodesAndAscendants(const OpGraph& op_graph, HashSet<OpNode*>* op_nodes) {
  std::list<OpNode*> starts;
  GetLossOpNodes(op_graph, &starts);
  auto ForEachNextNode = [&](OpNode* op_node, const std::function<void(OpNode*)>& Handler) {
    for (OpEdge* edge : op_node->in_edges()) {
      if (AnyLbiWithDiffLbi(edge)) { Handler(edge->src_node()); }
    }
  };
  op_graph.BfsForEachNode(starts, ForEachNextNode,
                          [&](OpNode* op_node) { op_nodes->emplace(op_node); });
}

std::function<bool(OpNode*)> MakePredicatorNeedBackwardOp(const OpGraph& op_graph) {
  auto var_op_nodes_and_descendants = std::make_shared<HashSet<OpNode*>>();
  GetVariableOpNodesAndDescendants(op_graph, var_op_nodes_and_descendants.get());
  auto loss_op_nodes_and_ascendants = std::make_shared<HashSet<OpNode*>>();
  GetLossOpNodesAndAscendants(op_graph, loss_op_nodes_and_ascendants.get());
  return [var_op_nodes_and_descendants, loss_op_nodes_and_ascendants](OpNode* op_node) {
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

void GenerateOriginDiffLbi(const LogicalBlobId& lbi, std::vector<OperatorConf>* op_confs,
                           LogicalBlobId* out_diff_lbi) {
  OperatorConf constant_like_op{};
  constant_like_op.set_name(lbi.op_name() + "_" + lbi.blob_name() + "_grad_ConstantLike");
  ConstantLikeOpConf* constant_like_conf = constant_like_op.mutable_constant_like_conf();
  constant_like_conf->set_like(GenLogicalBlobName(lbi));
  constant_like_conf->set_out("out");
  {
    int32_t origin_grad = GlobalJobDesc().loss_scale_factor();
    constant_like_conf->set_int_operand(origin_grad);
  }
  op_confs->push_back(constant_like_op);
  out_diff_lbi->set_op_name(constant_like_op.name());
  out_diff_lbi->set_blob_name(constant_like_conf->out());
}

ParallelConf ProducerParallelConf4Lbi(const OpGraph& op_graph, const LogicalBlobId& lbi) {
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
    OperatorConf scalar_mul_op_conf{};
    scalar_mul_op_conf.set_name("System-ModelDiffScale-ScalarMul_" + NewUniqueId());
    ScalarMulOpConf* scalar_mul_conf = scalar_mul_op_conf.mutable_scalar_mul_conf();
    scalar_mul_conf->set_float_operand(scale_factor);
    scalar_mul_conf->set_in(GenLogicalBlobName(diff_lbi));
    scalar_mul_conf->set_out("out");
    job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {scalar_mul_op_conf});
    diff_lbi.set_op_name(scalar_mul_op_conf.name());
    diff_lbi.set_blob_name(scalar_mul_conf->out());
  }
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
    parallel_conf.add_device_name("0:cpu:0");
    job_builder->AddOps(parallel_conf, {op_conf});

    total_loss_instance_num_lbi.set_op_name(op_conf.name());
    total_loss_instance_num_lbi.set_blob_name("out");
  } else {
    UNIMPLEMENTED();
  }
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    OperatorConf broadcast_div_op_conf{};
    broadcast_div_op_conf.set_name("System-ModelDiffScale-BroadcastDiv_" + NewUniqueId());
    BroadcastDivOpConf* broadcast_div_conf = broadcast_div_op_conf.mutable_broadcast_div_conf();
    broadcast_div_conf->set_a(GenLogicalBlobName(diff_lbi));
    broadcast_div_conf->set_b(GenLogicalBlobName(total_loss_instance_num_lbi));
    broadcast_div_conf->set_out("out");
    job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {broadcast_div_op_conf});
    diff_lbi.set_op_name(broadcast_div_op_conf.name());
    diff_lbi.set_blob_name(broadcast_div_conf->out());
  }
}

std::function<OpNode*(const std::string&)> MakeGetterLossOpNode4OpName(const OpGraph& op_graph) {
  std::list<OpNode*> loss_nodes;
  GetLossOpNodes(op_graph, &loss_nodes);
  auto loss_op_name2op_node = std::make_shared<HashMap<std::string, OpNode*>>();
  for (OpNode* op_node : loss_nodes) {
    CHECK(loss_op_name2op_node->emplace(op_node->op().op_name(), op_node).second);
  }
  return [loss_op_name2op_node](const std::string& op_name) -> OpNode* {
    return loss_op_name2op_node->at(op_name);
  };
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

void InitOutOba2OutDiffLbi(const std::list<OpNode*>& loss_nodes,
                           HashMap<OpBlobArg, LogicalBlobId>* out_oba2out_diff_lbi,
                           JobBuilder* job_builder) {
  for (const std::string& loss_lbn : GetTrainConf().loss_lbn()) {
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
    GenerateOriginDiffLbi(loss_lbi, &ops, out_diff_lbi);
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

void ClipGradientByGlobalNorm(const OpGraph& op_graph, JobBuilder* job_builder,
                              HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi,
                              const ClipByGlobalNormConf& conf) {
  if (lbi2diff_lbi->empty()) { return; }
  HashMap<LogicalBlobId, const ParallelDesc*> lbi2parallel_desc;
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    lbi2parallel_desc[lbi] = &model_op_node->parallel_desc();
  }
  bool all_same_parallel_desc = true;
  const ParallelDesc* parallel_desc = nullptr;
  for (const auto& pair : lbi2parallel_desc) {
    if (parallel_desc == nullptr) {
      parallel_desc = pair.second;
    } else if (*parallel_desc != *pair.second) {
      all_same_parallel_desc = false;
      break;
    } else {
      continue;
    }
  }
  ParallelConf global_norm_parallel_conf =
      all_same_parallel_desc ? parallel_desc->parallel_conf() : GenParallelConfOfCpuZeroOnMaster();
  OperatorConf global_norm_add_op_conf{};
  global_norm_add_op_conf.set_name("System-ClipGradient-GlobalNorm-Add-" + NewUniqueId());
  AddOpConf* global_norm_add_conf = global_norm_add_op_conf.mutable_add_conf();
  for (const auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& diff_lbi = pair.second;
    OperatorConf square_sum_op_conf{};
    square_sum_op_conf.set_name("System-ClipGradient-GlobalNorm-SquareSum-" + NewUniqueId());
    SquareSumOpConf* square_sum_conf = square_sum_op_conf.mutable_square_sum_conf();
    square_sum_conf->set_x(GenLogicalBlobName(diff_lbi));
    square_sum_conf->set_y("y");
    job_builder->AddOps(lbi2parallel_desc.at(pair.first)->parallel_conf(), {square_sum_op_conf});
    *global_norm_add_conf->mutable_in()->Add() =
        GenLogicalBlobName(square_sum_op_conf.name(), square_sum_conf->y());
  }
  global_norm_add_conf->set_out("out");
  job_builder->AddOps(global_norm_parallel_conf, {global_norm_add_op_conf});
  OperatorConf inv_global_norm_op_conf{};
  inv_global_norm_op_conf.set_name("System-ClipGradient-GlobalNorm-InvGlobalNorm-" + NewUniqueId());
  RsqrtOpConf* inv_global_norm_rsqrt_conf = inv_global_norm_op_conf.mutable_rsqrt_conf();
  inv_global_norm_rsqrt_conf->set_in(
      GenLogicalBlobName(global_norm_add_op_conf.name(), global_norm_add_conf->out()));
  inv_global_norm_rsqrt_conf->set_out("out");
  job_builder->AddOps(global_norm_parallel_conf, {inv_global_norm_op_conf});
  OperatorConf inv_clip_norm_op_conf{};
  inv_clip_norm_op_conf.set_name("System-ClipGradient-GlobalNorm-InvClipNorm-" + NewUniqueId());
  ConstantLikeOpConf* inv_clip_norm_constant_like_conf =
      inv_clip_norm_op_conf.mutable_constant_like_conf();
  inv_clip_norm_constant_like_conf->set_like(
      GenLogicalBlobName(inv_global_norm_op_conf.name(), inv_global_norm_rsqrt_conf->out()));
  inv_clip_norm_constant_like_conf->set_float_operand(1.0 / conf.clip_norm());
  inv_clip_norm_constant_like_conf->set_out("out");
  job_builder->AddOps(global_norm_parallel_conf, {inv_clip_norm_op_conf});
  OperatorConf minimum_op_conf{};
  minimum_op_conf.set_name("System-ClipGradient-GlobalNorm-Minimum-" + NewUniqueId());
  BroadcastMinimumOpConf* minimum_conf = minimum_op_conf.mutable_broadcast_minimum_conf();
  minimum_conf->set_a(
      GenLogicalBlobName(inv_global_norm_op_conf.name(), inv_global_norm_rsqrt_conf->out()));
  minimum_conf->set_b(
      GenLogicalBlobName(inv_clip_norm_op_conf.name(), inv_clip_norm_constant_like_conf->out()));
  minimum_conf->set_out("out");
  job_builder->AddOps(global_norm_parallel_conf, {minimum_op_conf});
  const std::string gradient_scale_factor_lbn =
      GenLogicalBlobName(minimum_op_conf.name(), minimum_conf->out());
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    OperatorConf scalar_mul_op_conf{};
    scalar_mul_op_conf.set_name("System-ClipGradient-GlobalNorm-ScalarMul-" + NewUniqueId());
    ScalarMulByTensorOpConf* scalar_mul_by_tensor_conf =
        scalar_mul_op_conf.mutable_scalar_mul_by_tensor_conf();
    scalar_mul_by_tensor_conf->set_in(GenLogicalBlobName(diff_lbi));
    scalar_mul_by_tensor_conf->set_scalar(gradient_scale_factor_lbn);
    scalar_mul_by_tensor_conf->set_out("out");
    job_builder->AddOps(lbi2parallel_desc.at(lbi)->parallel_conf(), {scalar_mul_op_conf});
    diff_lbi.set_op_name(scalar_mul_op_conf.name());
    diff_lbi.set_blob_name(scalar_mul_by_tensor_conf->out());
  }
}

}  // namespace

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

void GenerateBackwardOpConfWrapperStruct::Call(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) const {
  if (func_) {
    (*func_)(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
  } else if (naive_func_) {
    (*naive_func_)(op, op_confs, DiffLbi4BnInOp);
  } else {
    UNIMPLEMENTED();
  }
}

void GenerateBackwardOpConfIf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  std::unique_ptr<GenerateBackwardOpConfWrapperStruct> obj;
  obj.reset(NewObj<GenerateBackwardOpConfWrapperStruct>(op.op_conf().op_type_case()));
  obj->Call(op, op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
}

void AutoGrad(const OpGraph& op_graph, JobBuilder* job_builder,
              HashMap<LogicalBlobId, LogicalBlobId>* out_lbi2out_diff_lbi) {
  auto NeedBackwardOp = MakePredicatorNeedBackwardOp(op_graph);
  std::list<OpNode*> loss_nodes;
  GetLossOpNodes(op_graph, &loss_nodes);
  CheckNotReachableAmongOpNodes(op_graph, loss_nodes);
  for (OpNode* loss_node : loss_nodes) { CHECK(NeedBackwardOp(loss_node)); }

  // generate ones lbi as loss's diff
  HashMap<OpBlobArg, LogicalBlobId> out_oba2out_diff_lbi;
  InitOutOba2OutDiffLbi(loss_nodes, &out_oba2out_diff_lbi, job_builder);

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
  op_graph.TopoForEachNode(loss_nodes, ForEachOutNode, ForEachInNode, [&](OpNode* op_node) {
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
    };
    auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return op_graph.GetLogicalBlobDesc(op_node->op().BnInOp2Lbi(bn));
    };
    GenerateCloneGradOpIfNeed(*op_node, job_builder, in_oba2in_diff_lbi, &out_oba2out_diff_lbi,
                              &out_oba2clone_bw_add_out_lbi);
    std::vector<OperatorConf> ops;
    GenerateBackwardOpConfIf(op_node->op(), &ops, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
    job_builder->AddOps(op_node->parallel_desc().parallel_conf(), ops);
  });
  OpBlobArgPairs fw_bw_oba_pairs;
  CalcFwBwObaPairs(op_graph, in_oba2in_diff_lbi, out_oba2out_diff_lbi, out_oba2clone_bw_add_out_lbi,
                   *job_builder, &fw_bw_oba_pairs);
  BindFwBwObaPairs(op_graph, fw_bw_oba_pairs, job_builder);
  CalcOutLbi2OutDiffLbi(op_graph, out_oba2out_diff_lbi, out_lbi2out_diff_lbi);
}

void ScaleModelDiffByLossInstanceNum(const OpGraph& op_graph, JobBuilder* job_builder,
                                     HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  auto LossOpNode4OpName = MakeGetterLossOpNode4OpName(op_graph);
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
      // TODO: consider batch_axis or sbp
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
}

void ScaleModelDiffByLossScale(const OpGraph& op_graph, JobBuilder* job_builder,
                               HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  const int32_t loss_scale_factor = GlobalJobDesc().loss_scale_factor();
  if (loss_scale_factor == 1) { return; }
  const float down_scale_factor = 1.0 / loss_scale_factor;
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    OperatorConf down_scale_mul_op;
    down_scale_mul_op.set_name("System-ModelDiffScale-ScalarMul-" + NewUniqueId());
    ScalarMulOpConf* conf = down_scale_mul_op.mutable_scalar_mul_conf();
    conf->set_in(GenLogicalBlobName(diff_lbi));
    conf->set_out("out");
    conf->set_float_operand(down_scale_factor);
    job_builder->AddOps(ProducerParallelConf4Lbi(op_graph, lbi), {down_scale_mul_op});
    diff_lbi.set_op_name(down_scale_mul_op.name());
    diff_lbi.set_blob_name(conf->out());
  }
}

void RegularizeGradient(const OpGraph& op_graph, JobBuilder* job_builder,
                        HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi) {
  for (auto& pair : *lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    LogicalBlobId& diff_lbi = pair.second;
    const OpNode* model_op_node = op_graph.OpNode4OpName(lbi.op_name());
    CHECK(model_op_node->op().op_conf().has_variable_conf());
    const VariableOpConf& variable_conf = model_op_node->op().op_conf().variable_conf();
    if (!variable_conf.has_regularizer()) { continue; }
    const RegularizerConf& regularizer_conf = variable_conf.regularizer();
    if (regularizer_conf.has_l1_l2_conf()) {
      OperatorConf regularize_gradient_op_conf{};
      regularize_gradient_op_conf.set_name("System-RegularizeGradient-L1L2-" + NewUniqueId());
      L1L2RegularizeGradientOpConf* l1_l2_regularize_gradient_conf =
          regularize_gradient_op_conf.mutable_l1_l2_regularize_gradient_conf();
      l1_l2_regularize_gradient_conf->set_model(GenLogicalBlobName(lbi));
      l1_l2_regularize_gradient_conf->set_model_diff(GenLogicalBlobName(diff_lbi));
      l1_l2_regularize_gradient_conf->set_out("out");
      l1_l2_regularize_gradient_conf->set_l1(regularizer_conf.l1_l2_conf().l1());
      l1_l2_regularize_gradient_conf->set_l2(regularizer_conf.l1_l2_conf().l2());
      job_builder->AddOps(model_op_node->parallel_desc().parallel_conf(),
                          {regularize_gradient_op_conf});
      diff_lbi.set_op_name(regularize_gradient_op_conf.name());
      diff_lbi.set_blob_name(l1_l2_regularize_gradient_conf->out());
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
    OperatorConf parallel_cast_op_conf{};
    parallel_cast_op_conf.set_name("System-AutoGrad-ParallelCast-" + NewUniqueId());
    ParallelCastOpConf* parallel_cast_conf = parallel_cast_op_conf.mutable_parallel_cast_conf();
    parallel_cast_conf->set_in(GenLogicalBlobName(diff_lbi));
    parallel_cast_conf->set_out("out");
    const SbpParallel& model_sbp = model_op_node->SbpParallel4Lbi(lbi);
    if (model_sbp.has_broadcast_parallel()) {
      parallel_cast_conf->mutable_split_axis()->clear_value();
    } else if (model_sbp.has_split_parallel()) {
      parallel_cast_conf->mutable_split_axis()->set_value(model_sbp.split_parallel().axis());
    } else {
      UNIMPLEMENTED();
    }
    job_builder->AddOps(model_op_node->parallel_desc().parallel_conf(), {parallel_cast_op_conf});
    diff_lbi.set_op_name(parallel_cast_op_conf.name());
    diff_lbi.set_blob_name(parallel_cast_conf->out());
  }
}

}  // namespace oneflow
