#include "oneflow/core/job_completer/job_completer.h"
#include "oneflow/core/job_completer/autovar.h"
#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/job_completer/autotick.h"
#include "oneflow/core/job_completer/add_keep_header_only_op_conf.h"
#include "oneflow/core/job_completer/optimizer.h"
#include "oneflow/core/job_completer/add_saver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job_completer/all_reduce_add_pass.h"
#include "oneflow/core/job_completer/freeze_sbp_signature.h"
#include "oneflow/core/job_completer/aggregate_boxing_by_dst_parallel.h"

namespace oneflow {

namespace {

void WithOpGraphAndMutJob(Job* job, const std::function<void(const OpGraph&, Job*)>& Handler) {
  OpGraph op_graph(*job);
  Handler(op_graph, job);
}

void GenerateFacadeImplOpConfIf(const OpNode& op_node, JobBuilder* job_builder) {
  auto op_type_case = op_node.op().op_conf().op_type_case();
  if (IsClassRegistered<GenerateFacadeImplOpConfWrapperStruct>(op_type_case)) {
    auto* obj = NewObj<GenerateFacadeImplOpConfWrapperStruct>(op_type_case);
    obj->Call(op_node, job_builder);
  }
}

void ReplaceFacade(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  op_graph.ForEachNode(
      [&](OpNode* op_node) { GenerateFacadeImplOpConfIf(*op_node, &job_builder); });
}

void UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(
    const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi, Job* job) {
  auto& mut_pairs =
      (*job->mutable_helper()->mutable_tag2lbi_relations())[kProducedLbi2ConsumedDiffLbi];
  for (const auto& pair : lbi2diff_lbi) {
    auto* mut_pair = mut_pairs.add_pair();
    *mut_pair->mutable_first() = pair.first;
    *mut_pair->mutable_second() = pair.second;
  }
}

void BindIdenticalSbpObaPairsBetweenIbns(const OpNode& op_node, JobBuilder* job_builder) {
  HashMap<LogicalBlobId, std::vector<OpBlobArg>> in_lbi2obas;
  for (const std::string& ibn : op_node.op().input_bns()) {
    in_lbi2obas[op_node.op().BnInOp2Lbi(ibn)].push_back(GenOpBlobArg(op_node.op().op_name(), ibn));
  }
  for (const auto& pair : in_lbi2obas) {
    if (pair.second.size() > 1) {
      FOR_RANGE(int32_t, i, 1, pair.second.size()) {
        job_builder->BindIdenticalSbpOpBlobArgPair(pair.second.at(0), pair.second.at(i));
      }
    }
  }
}

void SetSbpSignatureHintByIdenticalSbpObaPairs(const OpGraph& op_graph, JobBuilder* job_builder) {
  HashMap<OpBlobArg, const SbpParallel*> oba2sbp_parallel;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto ForEachBn = [&](const std::function<void(const std::string&)>& Handler) {
      for (const auto& ibn : op_node->op().input_bns()) { Handler(ibn); }
      for (const auto& obn : op_node->op().output_bns()) { Handler(obn); }
    };
    ForEachBn([&](const std::string& bn_in_op) {
      const auto& oba = GenOpBlobArg(op_node->op().op_name(), bn_in_op);
      oba2sbp_parallel[oba] = &op_node->SbpParallel4Lbi(op_node->op().BnInOp2Lbi(bn_in_op));
    });
  });
  auto HasSbpParallel = [&](const OpBlobArg& oba) {
    return oba2sbp_parallel.find(oba) != oba2sbp_parallel.end();
  };
  for (const auto& pair : job_builder->job().helper().identical_sbp_oba_pairs().pair()) {
    const SbpParallel* sbp_parallel = nullptr;
    if (HasSbpParallel(pair.first()) && HasSbpParallel(pair.second())) {
      CHECK(oba2sbp_parallel.at(pair.first()) == oba2sbp_parallel.at(pair.second()));
      sbp_parallel = oba2sbp_parallel.at(pair.first());
    } else if (HasSbpParallel(pair.first())) {
      sbp_parallel = oba2sbp_parallel.at(pair.first());
    } else if (HasSbpParallel(pair.second())) {
      sbp_parallel = oba2sbp_parallel.at(pair.second());
    } else {
      UNIMPLEMENTED();
    }
    *job_builder->MutSbpParallel4Oba(pair.first()) = *sbp_parallel;
    *job_builder->MutSbpParallel4Oba(pair.second()) = *sbp_parallel;
  }
}

void UpdateOpSbpSignatureHint(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  op_graph.ForEachNode(
      [&](OpNode* op_node) { BindIdenticalSbpObaPairsBetweenIbns(*op_node, &job_builder); });
  SetSbpSignatureHintByIdenticalSbpObaPairs(op_graph, &job_builder);
}

void GenerateOpConf4Trainning(const OpGraph& op_graph, Job* job) {
  LogicalBlobId total_loss_instance_num;
  HashMap<LogicalBlobId, LogicalBlobId> lbi2diff_lbi;
  AutoGrad(op_graph, job, &lbi2diff_lbi);
  std::function<const LogicalBlobId&(const ParallelDesc&)> LossInstanceNum4ParallelDesc;
  AddTotalLossInstanceNumOpConf(op_graph, job, lbi2diff_lbi, &LossInstanceNum4ParallelDesc);
  AddOptimizerOpConf(op_graph, job, lbi2diff_lbi, LossInstanceNum4ParallelDesc);
  AddSaver(op_graph, job);
  UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(lbi2diff_lbi, job);
  UpdateOpSbpSignatureHint(op_graph, job);
}

std::function<ParallelConf*(const std::string&)> MakeGetterMutParallelConf4OpName(
    Placement* placement) {
  auto op_name2parallel_conf = std::make_shared<HashMap<std::string, ParallelConf*>>();
  FOR_RANGE(int, idx, 0, placement->placement_group_size()) {
    auto* placement_group = placement->mutable_placement_group(idx);
    for (const std::string& op_name : placement_group->op_set().op_name()) {
      ParallelConf* parallel_conf = placement_group->mutable_parallel_conf();
      CHECK(op_name2parallel_conf->emplace(op_name, parallel_conf).second);
    }
  }
  return [op_name2parallel_conf](const std::string& op_name) {
    return op_name2parallel_conf->at(op_name);
  };
}

std::function<OperatorConf*(const std::string&)> MakeMutableOperatorConf4OpName(Job* job) {
  auto op_name2op_conf = std::make_shared<HashMap<std::string, OperatorConf*>>();
  FOR_RANGE(int, idx, 0, job->net().op_size()) {
    OperatorConf* op_conf = job->mutable_net()->mutable_op(idx);
    CHECK(op_name2op_conf->emplace(op_conf->name(), op_conf).second);
  }
  return [op_name2op_conf](const std::string& op_name) { return op_name2op_conf->at(op_name); };
}

void AddIdentityOp(const std::string& prefix, Job* job, const HashSet<LogicalBlobId>& input_lbis,
                   HashMap<LogicalBlobId, LogicalBlobId>* old_lbi2new_lbi,
                   const ParallelConf& parallel_conf) {
  // add tuple identity op
  OperatorConf* tuple_identity_op = job->mutable_net()->add_op();
  tuple_identity_op->set_name(prefix + NewUniqueId());
  TupleIdentityOpConf* tuple_identity_op_conf = tuple_identity_op->mutable_tuple_identity_conf();
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : input_lbis) {
    std::string blob_name = std::string("out_") + std::to_string(idx++);
    {
      LogicalBlobId output_lbi;
      output_lbi.set_op_name(tuple_identity_op->name());
      output_lbi.set_blob_name(blob_name);
      CHECK(old_lbi2new_lbi->emplace(lbi, output_lbi).second);
    }
    tuple_identity_op_conf->add_in(lbi.op_name() + "/" + lbi.blob_name());
    tuple_identity_op_conf->add_out(blob_name);
  }
  // add placement of tuple identity op
  PlacementGroup* p_group = job->mutable_placement()->add_placement_group();
  *(p_group->mutable_op_set()->add_op_name()) = tuple_identity_op->name();
  *(p_group->mutable_parallel_conf()) = parallel_conf;
}

void AddIdentityOpAndReconnect(
    const std::string& identity_op_name_prefix, Job* job, const std::vector<OpEdge*>& op_edges,
    const std::function<OperatorConf*(const std::string&)>& MutOperatorConf4OpName,
    const ParallelConf& parallel_conf) {
  // add identity op
  HashSet<LogicalBlobId> lbis;
  for (OpEdge* edge : op_edges) { lbis.insert(edge->lbis().begin(), edge->lbis().end()); }
  HashMap<LogicalBlobId, LogicalBlobId> old_lbi2new_lbi;
  AddIdentityOp(identity_op_name_prefix, job, lbis, &old_lbi2new_lbi, parallel_conf);
  // reconnect to identity op
  for (OpEdge* edge : op_edges) {
    OperatorConf* op_conf = MutOperatorConf4OpName(edge->dst_node()->op().op_name());
    PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
    for (const LogicalBlobId& lbi : edge->lbis()) {
      std::string lbn_check = GenLogicalBlobName(lbi);
      std::string identity_out_lbn = GenLogicalBlobName(old_lbi2new_lbi.at(lbi));
      for (const std::string& ibn : edge->lbi2ibns().at(lbi)) {
        SetBnValInOpTypeConf(op_type_conf, ibn, lbn_check, identity_out_lbn);
      }
    }
  }
}

void FixTickOpIfExists(Job* job) {
  auto MutParallelConf4OpName = MakeGetterMutParallelConf4OpName(job->mutable_placement());
  OperatorConf* tick_op_conf = nullptr;
  FOR_RANGE(int, idx, 0, job->mutable_net()->op_size()) {
    OperatorConf* op_conf = job->mutable_net()->mutable_op(idx);
    if (op_conf->has_tick_conf()) {
      CHECK(tick_op_conf == nullptr);
      tick_op_conf = op_conf;
    }
  }
  if (tick_op_conf == nullptr) { return; }
  if (tick_op_conf->tick_conf().has_in()) { return; }
  std::map<OperatorConf::OpTypeCase, std::vector<OperatorConf*>> op_type_case2source_op_confs;
  FOR_RANGE(int, idx, 0, job->mutable_net()->op_size()) {
    OperatorConf* op_conf = job->mutable_net()->mutable_op(idx);
    if (op_conf == tick_op_conf) { continue; }
    DeviceType device_type = ParallelDesc(*MutParallelConf4OpName(op_conf->name())).device_type();
    if (ConstructOp(*op_conf, device_type)->input_bns().size() == 0) {
      op_type_case2source_op_confs[op_conf->op_type_case()].push_back(op_conf);
    }
  }
  if (op_type_case2source_op_confs.find(OperatorConf::kRecordLoadConf)
      != op_type_case2source_op_confs.end()) {
    CHECK_EQ(op_type_case2source_op_confs.size(), 1);
  }
  // set input of tick op
  OperatorConf* source_op_conf = op_type_case2source_op_confs.cbegin()->second.at(0);
  ParallelConf* source_parallel_conf = MutParallelConf4OpName(source_op_conf->name());
  DeviceType device_type = ParallelDesc(*source_parallel_conf).device_type();
  std::shared_ptr<Operator> source_op = ConstructOp(*source_op_conf, device_type);
  CHECK_GE(source_op->output_bns().size(), 1);
  LogicalBlobId src_first_output_lbi = source_op->BnInOp2Lbi(source_op->output_bns().Get(0));
  std::string source_op_output_lbn = GenLogicalBlobName(src_first_output_lbi);
  CHECK_EQ(tick_op_conf->tick_conf().has_in(), false);
  tick_op_conf->mutable_tick_conf()->set_in(source_op_output_lbn);
  // fix tick op placement
  *MutParallelConf4OpName(tick_op_conf->name()) = *source_parallel_conf;
  // add log_counter op connecting to tick op, making tick op always consumed
  OperatorConf* tick_log_counter = job->mutable_net()->add_op();
  tick_log_counter->set_name("tick_log_counter_" + NewUniqueId());
  LogCounterOpConf* tick_log_counter_conf = tick_log_counter->mutable_log_counter_conf();
  tick_log_counter_conf->set_in(tick_op_conf->name() + "/" + tick_op_conf->tick_conf().out());
  tick_log_counter_conf->set_interval(MaxVal<int32_t>::value);
  // add placement of tick_log_counter op
  PlacementGroup* p_group = job->mutable_placement()->add_placement_group();
  *(p_group->mutable_op_set()->add_op_name()) = tick_log_counter->name();
  *(p_group->mutable_parallel_conf()) = *source_parallel_conf;
}

void ConvertPseudoChainToChain(Job* job) {
  auto GetSourceNodesAndEdges = [&](const HashSet<OpNode*>& chain_nodes,
                                    HashSet<OpNode*>* source_nodes,
                                    std::vector<OpEdge*>* source_edges) {
    for (OpNode* node : chain_nodes) {
      for (OpEdge* edge : node->in_edges()) {
        if (chain_nodes.find(edge->src_node()) == chain_nodes.end()) {
          source_edges->push_back(edge);
          source_nodes->insert(node);
        }
      }
    }
  };
  auto MutOperatorConf4OpName = MakeMutableOperatorConf4OpName(job);
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job->placement());
  OpGraph(*job).ForEachPseudoChain([&](const HashSet<OpNode*>& chain_nodes) {
    HashSet<OpNode*> source_nodes;
    std::vector<OpEdge*> source_edges;
    GetSourceNodesAndEdges(chain_nodes, &source_nodes, &source_edges);
    if (source_edges.size() <= 1) { return; }
    if (source_nodes.size() <= 1) { return; }
    if (chain_nodes.size() - source_nodes.size() <= 2) { return; }
    const OpNode* first_node = *source_nodes.begin();
    if (first_node->parallel_desc().device_type() == DeviceType::kCPU) { return; }
    HashMap<bool, std::vector<OpEdge*>> has_diff2source_edges;
    for (OpEdge* edge : source_edges) { has_diff2source_edges[edge->has_diff()].push_back(edge); }
    for (const auto& pair : has_diff2source_edges) {
      HashSet<OpNode*> src_nodes;
      HashSet<OpNode*> dst_nodes;
      for (OpEdge* edge : pair.second) {
        src_nodes.emplace(edge->src_node());
        dst_nodes.emplace(edge->dst_node());
      }
      if (src_nodes.size() > 1 && dst_nodes.size() > 1) {
        AddIdentityOpAndReconnect("pseudo_chain_header_", job, pair.second, MutOperatorConf4OpName,
                                  *ParallelConf4OpName(first_node->op().op_name()));
      }
    }
  });
}

std::function<bool(OpNode*)> MakePredicatorIsReachableFromAnyVariableOps(const OpGraph& op_graph) {
  auto vars_and_decendants = std::make_shared<HashSet<OpNode*>>();
  GetVariableOpNodesAndDescendants(op_graph, vars_and_decendants.get());
  return [vars_and_decendants](OpNode* op_node) -> bool {
    return vars_and_decendants->find(op_node) != vars_and_decendants->end();
  };
}

void TieUpChainHeadersUnReachableFromAnyVariableOps(const OpGraph& op_graph, Job* job) {
  auto IsReachableFromAnyVariableOps = MakePredicatorIsReachableFromAnyVariableOps(op_graph);
  auto GetSourceNodesAndEdges = [&](const HashSet<OpNode*>& chain_nodes,
                                    std::vector<OpNode*>* source_nodes,
                                    std::vector<OpEdge*>* source_edges) {
    for (OpNode* node : chain_nodes) {
      for (OpEdge* edge : node->in_edges()) {
        if (chain_nodes.find(edge->src_node()) == chain_nodes.end()
            && IsReachableFromAnyVariableOps(edge->src_node()) == false) {
          source_edges->push_back(edge);
          source_nodes->push_back(node);
        }
      }
    }
  };
  auto MutOperatorConf4OpName = MakeMutableOperatorConf4OpName(job);
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job->placement());
  op_graph.ForEachChainFamily([&](const HashSet<OpNode*>& chain_nodes) {
    std::vector<OpNode*> source_nodes;
    std::vector<OpEdge*> source_edges;
    GetSourceNodesAndEdges(chain_nodes, &source_nodes, &source_edges);
    if (source_edges.size() <= 1) { return; }
    if (source_nodes.size() <= 1) { return; }
    // ignore small chain
    if (chain_nodes.size() - source_nodes.size() <= 2) { return; }
    AddIdentityOpAndReconnect("pseudo_chain_header_", job, source_edges, MutOperatorConf4OpName,
                              *ParallelConf4OpName(source_nodes.at(0)->op().op_name()));
  });
}

void AddIdentityOpForAllReduceOverlapingUntrainble(Job* job) {
  auto MutOperatorConf4OpName = MakeMutableOperatorConf4OpName(job);
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job->placement());
  OpGraph(*job).TopoForEachNode([&](OpNode* op_node) {
    if (op_node->HasBackward()) { return; }
    HashMap<bool, std::vector<OpEdge*>> has_bw2out_op_edges;
    for (OpEdge* edge : op_node->out_edges()) {
      has_bw2out_op_edges[edge->dst_node()->HasBackward()].push_back(edge);
    }
    if (has_bw2out_op_edges.size() <= 1) { return; }
    // only handle op_nodes that:
    // a) have no backward node;
    // b) have trainable and untrainble consumers;

    // group out_edge by trainable consumers' ParallelDesc
    HashMap<ParallelDesc, std::vector<OpEdge*>> consumer_op_pr2edges;
    for (OpEdge* edge : has_bw2out_op_edges.at(true)) {
      ParallelDesc pr(*ParallelConf4OpName(edge->dst_node()->op().op_name()));
      consumer_op_pr2edges[pr].push_back(edge);
    }
    for (const auto& pair : consumer_op_pr2edges) {
      AddIdentityOpAndReconnect(
          "all_reduce_overlapping_untrainable_", job, pair.second, MutOperatorConf4OpName,
          *ParallelConf4OpName(pair.second.at(0)->dst_node()->op().op_name()));
    }
  });
}

void FixAndOptimizeDLNet(Job* job) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  if (!(job_desc->IsPredict()
        && job_desc->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf())) {
    FixTickOpIfExists(job);
    // ConvertPseudoChainToChain(job);
  }
  // if (job_desc->IsTrain()) { AddIdentityOpForAllReduceOverlapingUntrainble(job); }
}

void SetOpTimeShape(const OpGraph& op_graph, Job* job) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto* op_time_shape =
        &(*job->mutable_helper()->mutable_op_name2op_time_shape())[op_node->op().op_name()];
    if (op_node->out_blob_time_shape() != nullptr) {
      op_node->out_blob_time_shape()->ToProto(op_time_shape->mutable_out_blob_time_shape());
    }
    const auto* in_blob_fastest_time_shape = op_node->GetInputBlobFastestTimeShape();
    if (in_blob_fastest_time_shape != nullptr) {
      in_blob_fastest_time_shape->ToProto(op_time_shape->mutable_in_blob_fastest_time_shape());
    }
  });
}

void SetCtrlInOpName(const OpGraph& op_graph, Job* job) {
  auto IsMutableConsumedLbi = [](const Operator& op, const LogicalBlobId& lbi) -> bool {
    for (const std::string& bn : op.input_bns()) {
      if (op.BnInOp2Lbi(bn) == lbi && op.InputBlobModifier4Ibn(bn).is_mutable()) { return true; }
    }
    return false;
  };
  auto IsMdSaveEveryNthOp = [](const OperatorConf& op_conf) -> bool {
    return op_conf.has_every_nth_conf();
  };
  JobBuilder job_builder(job);
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_variable_conf() == false) { return; }
    if (op_node->out_edges().size() <= 1) { return; }
    const Operator& variable_op = op_node->op();
    const LogicalBlobId& variable_lbi = variable_op.BnInOp2Lbi(variable_op.SoleObn());
    const OperatorConf* mutable_consumer = nullptr;
    const OperatorConf* model_save_every_nth_op_conf = nullptr;
    std::vector<const OperatorConf*> naive_consumers;
    for (OpEdge* edge : op_node->out_edges()) {
      const auto& op_conf = edge->dst_node()->op().op_conf();
      if (IsMutableConsumedLbi(edge->dst_node()->op(), variable_lbi)) {
        CHECK(mutable_consumer == nullptr);
        mutable_consumer = &op_conf;
      } else if (IsMdSaveEveryNthOp(op_conf)) {
        CHECK(model_save_every_nth_op_conf == nullptr);
        model_save_every_nth_op_conf = &op_conf;
      } else {
        naive_consumers.push_back(&op_conf);
      }
    }
    if (mutable_consumer == nullptr) { return; }
    OperatorConf mut_mutable_consumer_op_conf(*mutable_consumer);
    for (const auto* fw_bw_op : naive_consumers) {
      mut_mutable_consumer_op_conf.add_ctrl_in_op_name(fw_bw_op->name());
    }
    job_builder.MutOps({mut_mutable_consumer_op_conf});
    if (model_save_every_nth_op_conf != nullptr) {
      OperatorConf mut_model_save_every_nth_op_conf(*model_save_every_nth_op_conf);
      mut_model_save_every_nth_op_conf.add_ctrl_in_op_name(mutable_consumer->name());
      job_builder.MutOps({mut_model_save_every_nth_op_conf});
    }
  });
}

void SetBatchDimLbis(const OpGraph& op_graph, Job* job) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(obn);
      if (op_node->HasBatchDim4Lbi(lbi)) {
        *job->mutable_helper()->mutable_batch_dim_lbis()->Add() = lbi;
      }
    }
  });
}

void SetOpTimeShape7CtrlInOpName7ModelLbis(const OpGraph& op_graph, Job* job) {
  SetOpTimeShape(op_graph, job);
  SetCtrlInOpName(op_graph, job);
  SetBatchDimLbis(op_graph, job);
}

void RewriteBoxingWithAllReduce(const OpGraph& op_graph, Job* job) {
  AllReduceAddPass().Apply(op_graph, job);
}

}  // namespace

void JobCompleter::Complete(Job* job) const {
  // replace facade op
  WithOpGraphAndMutJob(job, &ReplaceFacade);
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    // complete variable ops
    WithOpGraphAndMutJob(job, &AutoVar);
    WithOpGraphAndMutJob(job, &TieUpChainHeadersUnReachableFromAnyVariableOps);
    // complete ops for trainning
    HashMap<std::string, HashMap<std::string, LogicalBlobId>> op_name2ibn2in_diff_lbi;
    WithOpGraphAndMutJob(job, &GenerateOpConf4Trainning);
    // complete tick ops
    WithOpGraphAndMutJob(job, &AutoTick);
    // add keep_header_only op
    WithOpGraphAndMutJob(job, &RewriteBoxingWithAllReduce);
    WithOpGraphAndMutJob(job, &FreezeSbpSignature);
    WithOpGraphAndMutJob(job, &AggregateBoxingByDstParallel);
    WithOpGraphAndMutJob(job, &AddKeepHeaderOnlyOp);
    WithOpGraphAndMutJob(job, &SetOpTimeShape7CtrlInOpName7ModelLbis);
  }
  // TODO: refine
  FixAndOptimizeDLNet(job);
}

}  // namespace oneflow
