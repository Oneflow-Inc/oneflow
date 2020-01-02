#include "oneflow/core/job_completer/job_completer.h"
#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/job_completer/autotick.h"
#include "oneflow/core/job_completer/add_keep_header_only_op_conf.h"
#include "oneflow/core/job_completer/optimizer.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job_completer/all_reduce_add_pass.h"
#include "oneflow/core/job_completer/set_default_variable_conf.h"
#include "oneflow/core/job_completer/all_reduce_sequence_pass.h"
#include "oneflow/core/job_completer/group_boxing_by_dst_parallel.h"
#include "oneflow/core/job_completer/auto_mixed_precision.h"
#include "oneflow/core/job_completer/non_distributed_optimizer_pass.h"
#include "oneflow/core/job_completer/nccl_tuple_broadcast_reduce_sequence_pass.h"
#include "oneflow/core/job_completer/auto_train_step.h"
#include "oneflow/core/job_completer/auto_learning_rate.h"
#include "oneflow/core/job_completer/add_lbi_diff_watcher.h"
#include "oneflow/core/framework/config_def.h"

#include "oneflow/core/job_completer/xrt_compilation.h"

namespace oneflow {

namespace {

void CheckOpGraph(const OpGraph& op_graph) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    size_t in_cnt = 0;
    op_graph.ForEachDataAndCtrlInNode(op_node, [&](OpNode*) { ++in_cnt; });
    if (in_cnt == 0) { CHECK(op_node->op().op_conf().has_source_tick_conf()); }
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt == 0) { CHECK(op_node->op().op_conf().has_sink_tick_conf()); }
  });
}

void WithOpGraphAndMutJob(Job* job, const std::function<void(const OpGraph&, Job*)>& Handler) {
  OpGraph op_graph(*job);
  Handler(op_graph, job);
}

void WithOpGraphAndMutJobBuilder(Job* job,
                                 const std::function<void(const OpGraph&, JobBuilder*)>& Handler) {
  OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  Handler(op_graph, &job_builder);
}

void GenerateFacadeImplOpConfIf(const OpNode& op_node, JobBuilder* job_builder) {
  auto op_type_case = op_node.op().op_conf().op_type_case();
  if (IsClassRegistered<GenerateFacadeImplOpConfWrapperStruct>(op_type_case)) {
    std::unique_ptr<GenerateFacadeImplOpConfWrapperStruct> obj;
    obj.reset(NewObj<GenerateFacadeImplOpConfWrapperStruct>(op_type_case));
    obj->Call(op_node, job_builder);
  }
}

void ReplaceFacade(const OpGraph& op_graph, JobBuilder* job_builder) {
  op_graph.ForEachNode([&](OpNode* op_node) { GenerateFacadeImplOpConfIf(*op_node, job_builder); });
}

void UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(
    const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi, JobBuilder* job_builder) {
  auto& mut_pairs =
      (*job_builder->mutable_helper()->mutable_tag2lbi_relations())[kProducedLbi2ConsumedDiffLbi];
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

void UpdateOpSbpSignatureHint(const OpGraph& op_graph, JobBuilder* job_builder) {
  op_graph.ForEachNode(
      [&](OpNode* op_node) { BindIdenticalSbpObaPairsBetweenIbns(*op_node, job_builder); });
  SetSbpSignatureHintByIdenticalSbpObaPairs(op_graph, job_builder);
}

void GenerateOpConf4Trainning(const OpGraph& op_graph, JobBuilder* job_builder) {
  LogicalBlobId total_loss_instance_num;
  HashMap<LogicalBlobId, LogicalBlobId> lbi2diff_lbi;
  AutoGrad(op_graph, job_builder, &lbi2diff_lbi);
  std::function<const LogicalBlobId&(const ParallelDesc&)> LossInstanceNum4ParallelDesc;
  AddTotalLossInstanceNumOpConf(op_graph, job_builder, lbi2diff_lbi, &LossInstanceNum4ParallelDesc);
  AddOptimizerOpConf(op_graph, job_builder, lbi2diff_lbi, LossInstanceNum4ParallelDesc);
  UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(lbi2diff_lbi, job_builder);
  UpdateOpSbpSignatureHint(op_graph, job_builder);
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

void AddIdentityOp(const std::string& op_name, Job* job, const HashSet<LogicalBlobId>& input_lbis,
                   HashMap<LogicalBlobId, LogicalBlobId>* old_lbi2new_lbi,
                   HashMap<LogicalBlobId, std::string>* old_lbi2new_obn,
                   const ParallelConf& parallel_conf) {
  // add tuple identity op
  OperatorConf* tuple_identity_op = job->mutable_net()->add_op();
  tuple_identity_op->set_name(op_name);
  TupleIdentityOpConf* tuple_identity_op_conf = tuple_identity_op->mutable_tuple_identity_conf();
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : input_lbis) {
    const std::string& obn = GenRepeatedBn("out", idx++);
    const std::string& blob_name = obn;
    {
      LogicalBlobId output_lbi;
      output_lbi.set_op_name(tuple_identity_op->name());
      output_lbi.set_blob_name(blob_name);
      CHECK(old_lbi2new_lbi->emplace(lbi, output_lbi).second);
      CHECK(old_lbi2new_obn->emplace(lbi, obn).second);
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
    const std::string& op_name_prefix, Job* job, const std::vector<OpEdge*>& op_edges,
    const std::function<OperatorConf*(const std::string&)>& MutOperatorConf4OpName,
    const ParallelConf& parallel_conf) {
  // add identity op
  HashSet<LogicalBlobId> lbis;
  for (OpEdge* edge : op_edges) { lbis.insert(edge->lbis().begin(), edge->lbis().end()); }
  HashMap<LogicalBlobId, LogicalBlobId> old_lbi2new_lbi;
  HashMap<LogicalBlobId, std::string> old_lbi2new_obn;
  const auto& identity_op_name = op_name_prefix + NewUniqueId();
  AddIdentityOp(identity_op_name, job, lbis, &old_lbi2new_lbi, &old_lbi2new_obn, parallel_conf);
  // reconnect to identity op
  HashMap<LogicalBlobId, SbpParallel> old_lbi2sbp_parallel;
  for (OpEdge* edge : op_edges) {
    OperatorConf* op_conf = MutOperatorConf4OpName(edge->dst_node()->op().op_name());
    PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
    for (const LogicalBlobId& lbi : edge->lbis()) {
      std::string lbn_check = GenLogicalBlobName(lbi);
      std::string identity_out_lbn = GenLogicalBlobName(old_lbi2new_lbi.at(lbi));
      for (const std::string& ibn : edge->lbi2ibns().at(lbi)) {
        ReplaceInputLbnInOpCustomizedConf(op_type_conf, ibn, lbn_check, identity_out_lbn);
        const auto& sbp_parallel = edge->dst_node()->SbpParallel4BnInOp(ibn);
        const auto& sbp_iter = old_lbi2sbp_parallel.find(lbi);
        if (sbp_iter == old_lbi2sbp_parallel.end()) {
          old_lbi2sbp_parallel[lbi] = sbp_parallel;
        } else {
          CHECK(sbp_iter->second == sbp_parallel);
        }
      }
    }
  }
  // set sbp conf for tuple_identity_op
  CHECK_EQ(old_lbi2new_obn.size(), old_lbi2sbp_parallel.size());
  auto* sbp_sig_conf_map = job->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf();
  auto* sbp_parallel_map = (*sbp_sig_conf_map)[identity_op_name].mutable_bn_in_op2sbp_parallel();
  for (const auto& pair : old_lbi2new_obn) {
    (*sbp_parallel_map)[pair.second] = old_lbi2sbp_parallel.at(pair.first);
  }
}

std::function<bool(OpNode*)> MakePredicatorIsReachableFromAnyVariableOps(const OpGraph& op_graph) {
  auto vars_and_decendants = std::make_shared<HashSet<OpNode*>>();
  GetVariableOpNodesAndDescendants(op_graph, vars_and_decendants.get());
  return [vars_and_decendants](OpNode* op_node) -> bool {
    return vars_and_decendants->find(op_node) != vars_and_decendants->end();
  };
}

REGISTER_FUNCTION_CONFIG_DEF().Bool("enable_pseudo_chain_merge", false,
                                    "ties up chain headers unreachable from any variable ops");
class TieUpChainHeadersUnReachableFromAnyVariableOps final : public OpGraphPass {
  bool IsEnabled() const override { return GlobalJobDesc().Bool("enable_pseudo_chain_merge"); }
  void Apply(const OpGraph& op_graph, Job* job) override {
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
};

void SetCtrlInOpName4VariableOp(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto IsMutableConsumedLbi = [](const Operator& op, const LogicalBlobId& lbi) -> bool {
    for (const std::string& bn : op.input_bns()) {
      if (op.BnInOp2Lbi(bn) == lbi && op.InputBlobModifier4Ibn(bn).is_mutable()) { return true; }
    }
    return false;
  };
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_variable_conf() == false) { return; }
    if (op_node->out_edges().size() <= 1) { return; }
    const Operator& variable_op = op_node->op();
    const LogicalBlobId& variable_lbi = variable_op.BnInOp2Lbi(variable_op.SoleObn());
    const OperatorConf* mutable_consumer = nullptr;
    std::vector<const OperatorConf*> naive_consumers;
    for (OpEdge* edge : op_node->out_edges()) {
      const auto& op_conf = edge->dst_node()->op().op_conf();
      if (IsMutableConsumedLbi(edge->dst_node()->op(), variable_lbi)) {
        CHECK(mutable_consumer == nullptr);
        mutable_consumer = &op_conf;
      } else {
        naive_consumers.push_back(&op_conf);
      }
    }
    if (mutable_consumer == nullptr) { return; }
    OperatorConf mut_mutable_consumer_op_conf(*mutable_consumer);
    for (const auto* fw_bw_op : naive_consumers) {
      mut_mutable_consumer_op_conf.add_ctrl_in_op_name(fw_bw_op->name());
    }
    job_builder->MutOpsOnlyOnce({mut_mutable_consumer_op_conf});
  });
}

void SetOpTimeShape7BatchAxisLbis(const OpGraph& op_graph, JobBuilder* job_builder) {
  op_graph.DumpOpTimeShape(job_builder);
  op_graph.DumpBatchAxisLbi(job_builder);
}

void DumpLogicalBlobDescAndSbpSignature(const OpGraph& op_graph, JobBuilder* job_builder) {
  op_graph.DumpLogicalBlobDesc(job_builder);
  op_graph.DumpSbpSignature(job_builder);
}

void MakeNcclTupleBroadcastReduceSequence(const OpGraph& op_graph, JobBuilder* job_builder) {
  NcclTupleBroadcastReduceSequencePass().Apply(op_graph, job_builder);
}

}  // namespace

void JobCompleter::Complete(Job* job) const {
  // replace facade op
  WithOpGraphAndMutJobBuilder(job, &ReplaceFacade);
  // complete variable ops
  WithOpGraphAndMutJobBuilder(job, &SetDefaultVariableConf);
  AutoMixedPrecision()(job);
  if (GlobalJobDesc().IsTrain()) {
    TieUpChainHeadersUnReachableFromAnyVariableOps()(job);
    NonDistributedOptimizerPass()(job);
    WithOpGraphAndMutJob(job, &AutoTrainStep);
    WithOpGraphAndMutJob(job, &AutoLearningRate);
    // complete ops for trainning
    WithOpGraphAndMutJobBuilder(job, &GenerateOpConf4Trainning);
    WithOpGraphAndMutJobBuilder(job, &MakeNcclTupleBroadcastReduceSequence);
    AllReduceAddPass()(job);
    AddLbiDiffWatcherOpConfs(job);
    AllReduceSequencePass()(job);
  }
  WithOpGraphAndMutJobBuilder(job, &DumpLogicalBlobDescAndSbpSignature);
  WithOpGraphAndMutJobBuilder(job, &GroupBoxingByDstParallel);
  WithOpGraphAndMutJobBuilder(job, &AddKeepHeaderOnlyOp);
  WithOpGraphAndMutJobBuilder(job, &SetCtrlInOpName4VariableOp);
  // complete tick ops
  WithOpGraphAndMutJobBuilder(job, &AutoSourceTick);
  WithOpGraphAndMutJobBuilder(job, &AddTickForTimeShape);
  WithOpGraphAndMutJobBuilder(job, &AutoSinkTick);
  AddGlobalTotalJobCriticalSection(*job);
  WithOpGraphAndMutJobBuilder(job, &AddGlobalInputCriticalSections);
  WithOpGraphAndMutJobBuilder(job, &AddGlobalOutputCriticalSections);
  WithOpGraphAndMutJobBuilder(job, &DumpLogicalBlobDescAndSbpSignature);
  WithOpGraphAndMutJobBuilder(job, &SetOpTimeShape7BatchAxisLbis);

  if (XrtCompilationEnabled(GlobalJobDesc())) {
#ifdef OF_WITH_XRT
    WithOpGraphAndMutJob(job, &RebuildXrtCompiledJob);
#else
    LOG(WARNING) << "It will not use XLA or TensorRT since WITH_XLA or "
                    "WITH_TENSORRT was not enabled when compiling the project.";
#endif  // OF_WITH_XRT
  }
  CheckOpGraph(OpGraph(*job));
}

}  // namespace oneflow
