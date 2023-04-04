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

#include "oneflow/core/auto_parallel/sbp_constructor.h"
#include "oneflow/core/auto_parallel/auto_memory.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/auto_parallel/sbp_collector.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace auto_parallel {

namespace {

// AMS, a.k.a. Applied Mathematics & Statistics, is a department of the Stony Brook University.
// It contains 5 tracks: Computational & Applied Mathematics, Computational Biology,
// Operation Research, Quantitative Finance, Statistics.
AutoMemoryStrategy ams;

// kMemoryRatio increase by this rate at each time.
static const double kMemoryIncreaseRatio = 2.0;
// The ceil of kMemoryRatio.
static const double kMaxMemoryRatio = 22.0;
// The floor of kMemoryRatio
static const double kMinMemoryRatio = 0.1;
// If the current memory > available memory * kImpossibleRatio,
// then it is impossible to reduce the memory to an acceptable size
static const double kImpossibleRatio = 1.4;

// Pick from 5 fixed types of memory ratio.
double UpdateMemoryRatio() {
  switch (ams) {
    case kAdaptiveAutoMemory:
    case kDisableAutoMemory: return 0.0;
    case kSlightAutoMemory: return 0.4;
    case kModerateAutoMemory: return 4.3;
    default: return 11.0;  // case kHeavyAutoMemory
  }
}

}  // namespace

double kMemoryRatio;

Maybe<void> SbpConstructor::Init(const OpGraph& op_graph, Job* job /*Maybe not use*/) {
  JUST(InitSbpGraph(op_graph, *job));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitSbpGraph(const OpGraph& op_graph, const Job& job) {
  // Update nccl_use_compute_stream
  nccl_use_compute_stream_ = Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream();
  ams = job.job_conf().enable_auto_memory();
  kMemoryRatio = UpdateMemoryRatio();
  // TODO: process local node
  JUST(GenerateNodeAndEdge(op_graph, job));
  JUST(FillSbpSignatureForOpNode(op_graph, job));
  JUST(InitComputationCost(op_graph));
  if (enable_trunk_algo_) { JUST(ApplyTrunkAlgo()); }
  // Load logical blobs on all sbp edges.
  LoadLbi2SbpEdge(op_graph);
  // InitMemory() should be run before the sbp collector and after the ApplyTrunkAlgo() and
  // LoadLbi2SbpEdge(op_graph).
  InitAvailableMemory();
  InitMemory(op_graph, &sbp_graph_, nccl_use_compute_stream_);
  if (use_sbp_collector_) {
    // Use sbp collector to create sbp proxy for nodes with multiple downstream operators.
    SbpCollector sbp_collector;
    sbp_collector.CollectUniverse(sbp_graph_);
    // TODO: Init memory cost for proxy
    sbp_collector.ProxySbpCandidate(op_graph, op_name2sbp_node_, sbp_graph_);
  }

  JUST(InitCopyAndMemoryCost(op_graph));
  // We need to store the original cost and memory after the initialization (InitComputationCost(),
  // InitMemory(), InitCopyAndMemoryCost()) and before the usage of them (InitWeightedCost())
  sbp_graph_.StoreOriginMemory();
  InitWeightedCost();
  // TODO:  Set all the sbp signature id to be 0 for initialization.
  //        Could revert it back to
  // sbp_graph_.RandomSbpSignature(use_sbp_collector_);
  //        after settling down the synchronization of sbp strategy.
  sbp_graph_.SetDefaultSbpSig();
  double ori_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Initial cost: " << ori_cost;
  // If we do not prune those parallel cast ops, steal the initial strategy from user setting and
  // semi-auto parallelism
  if (!job.job_conf().enable_auto_parallel_ignore_user_sbp_config()) {
    JUST(StealSbpSignatureFromOpNode(op_graph, job));
    ori_cost = sbp_graph_.ComputeCost();
    LOG(INFO) << "OpGraph cost: " << ori_cost;
  }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::FindBestSbpSignature() {
  double ori_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Initial cost: " << ori_cost;
  int elimination_num = sbp_graph_.NodeAndEdgeEliminations();
  LOG(INFO) << "Elimination number: " << elimination_num;
  if (ori_cost > GetValidMaxCopyCost()) {
    JUST(sbp_graph_.Find1Strategy4Greedy());
    ori_cost = sbp_graph_.ComputeCost();
    LOG(INFO) << "Greedy cost: " << ori_cost;
  }

  int32_t step = 1;
  while (true) {
    sbp_graph_.GreedyStrategy(/*nbh_num=*/4);
    double curr_memory = sbp_graph_.GetMemory();
    double total_weighted_cost = sbp_graph_.ComputeWeightedCost();
    LOG(INFO) << "The " << step << "-th try, memory ratio: " << kMemoryRatio
              << ", memory: " << curr_memory << ", total cost: " << total_weighted_cost
              << ", time cost: " << (total_weighted_cost - kMemoryRatio * curr_memory);
    if (ams != AutoMemoryStrategy::kAdaptiveAutoMemory) { break; }
    if (curr_memory < available_memory_ || kMemoryRatio >= kMaxMemoryRatio) { break; }
    if (curr_memory > available_memory_ * kImpossibleRatio) {
      kMemoryRatio = kMaxMemoryRatio;
    } else {
      kMemoryRatio =
          std::max(std::min(kMaxMemoryRatio, kMemoryRatio * kMemoryIncreaseRatio), kMinMemoryRatio);
    }
    step++;
    sbp_graph_.ReComputeWeightedCost();
  }
  sbp_graph_.FinalizeSbp();

  double final_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Final cost: " << final_cost;
  // TODO: Restart searching with another original random strategy
  CHECK_LT_OR_RETURN(final_cost, GetValidMaxCopyCost())
      << "Failed! Auto parallel can't find a strategy with reasonable cost!";
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::DumpNdSbpSignatureForJob(const OpGraph& op_graph, Job* job) {
  for (auto& op_conf : *job->mutable_net()->mutable_op()) {
    const OpNode* node = op_graph.OpNode4OpName(op_conf.name());
    SbpNode* sbp_node = op_name2sbp_node_[node->op().op_name()];
    const NdSbpSignature& nd_sbp_sig = sbp_node->FinalSbpSignature();
    // Update NdSbpSignature
    (*job->mutable_job_parallel_view_conf()
          ->mutable_op_name2nd_sbp_signature_conf())[node->op().op_name()]
        .CopyFrom(nd_sbp_sig);
    // If we have 1D SbpSignature Conf
    if (node->parallel_desc().hierarchy()->NumAxes() == 1) {
      // Update SbpSignature
      SbpSignature sbp_signature;
      NdSbpSignatureToSbpSignature(nd_sbp_sig, &sbp_signature);
      (*job->mutable_job_parallel_view_conf()
            ->mutable_op_name2sbp_signature_conf())[node->op().op_name()]
          .CopyFrom(sbp_signature);
    }
    JUST(node->op().GetDumpNdSbpSignatureForOpConfFn()(nd_sbp_sig, &op_conf));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::GenerateNodeAndEdge(const OpGraph& op_graph, const Job& job) {
  JobParallelViewConf job_parallel_view_conf(job.job_parallel_view_conf());

  // Collect op_node
  std::vector<OpNode*> op_node_list;
  op_graph.ForEachNode([&](OpNode* op_node) {
    // TODO: support local op
    bool is_local_conf = false;
    {
      const auto& op_name2is_local = job_parallel_view_conf.op_name2is_local_parallel_view();
      const auto& iter = op_name2is_local.find(op_node->op().op_name());
      if (iter != op_name2is_local.end()) { is_local_conf = iter->second; }
    }
    CHECK(is_local_conf == false) << "Haven't deal with local operators.";
    op_node_list.push_back(op_node);
  });

  // Decide the order to visit the op
  std::vector<int32_t> order;
  auto CompareOpName = [&](OpNode* a, OpNode* b) {
    return a->op().op_name().compare(b->op().op_name()) > 0;
  };
  auto_parallel::DecideOrder(op_node_list, order, CompareOpName);
  std::vector<int32_t> output_order;

  // Create sbp nodes
  for (int32_t i = 0; i < op_node_list.size(); i++) {
    OpNode* op_node = op_node_list[order[i]];
    // Generate sbp node in cost model and link it with corresponding op node
    SbpNode* sbp_node = sbp_graph_.GenerateNode();
    // Mapping from sbp_node to op_node
    sbp_node->op_node_ = op_node;  // TODO: SetOpNode()
    op_name2sbp_node_[op_node->op().op_name()] = sbp_node;
  }
  // Create sbp edges
  for (int32_t i = 0; i < op_node_list.size(); i++) {
    OpNode* op_node = op_node_list[order[i]];
    // Get corresponding sbp node
    SbpNode* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    std::vector<OpNode*> output_node_list;
    for (const auto* op_edge : op_node->out_edges()) {
      output_node_list.push_back(op_edge->dst_node());
    }
    auto_parallel::DecideOrder(output_node_list, output_order, CompareOpName);
    for (int32_t j : output_order) {
      const auto& end_node_name = output_node_list[j]->op().op_name();
      // Generate sbp edge in cost model
      sbp_node->PointTo(op_name2sbp_node_[end_node_name]);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::FillSbpSignatureForOpNode(const OpGraph& op_graph, const Job& job) {
  // TODO: use user sbp signature in JobParallelViewConf
  // const JobParallelViewConf& job_parallel_view_conf(job.job_parallel_view_conf());
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    HashMap<std::string, const BlobDesc*> ibn2blob_desc;
    auto FindShape4Blobs = [&](const PbRpf<std::string>& bns) -> Maybe<void> {
      for (const std::string& ibn : bns) {
        const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
        const BlobDesc* logical_blob_desc = &op_node->LogicalBlobDesc4Lbi(lbi);
        ibn2blob_desc.emplace(ibn, logical_blob_desc);
      }
      return Maybe<void>::Ok();
    };
    JUST(FindShape4Blobs(op_node->op().input_bns()));
    JUST(FindShape4Blobs(op_node->op().output_bns()));
    // Get logical blob description
    auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
      const auto& it = ibn2blob_desc.find(ibn);
      if (it == ibn2blob_desc.end()) {
        return Error::InvalidValueError()
               << "Cannot find corresponding blob description for input_blob_name : " + ibn + " in "
                      + op_node->op().op_name();
      }
      return *(it->second);
    };
    // Get all valid sbp_signatures
    SbpNode* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    JUST(op_node->op().GetValidNdSbpSignatureList(LogicalBlobDesc4Ibn, op_node->parallel_desc(),
                                                  &sbp_node->sbp_sig_list_, /*check_output=*/true));
    sbp_node->InitializeSbp();
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::StealSbpSignatureFromOpNode(const OpGraph& op_graph, const Job& job) {
  // Steal some strategy from original op graph
  for (auto* sbp_node : sbp_graph_.node_list_) {
    // sbp_collectors do not have op_node
    if (sbp_node->op_node_) {
      for (int32_t sbp_id = 0; sbp_id < sbp_node->sbp_sig_list_.size(); sbp_id++) {
        if (*JUST(sbp_node->op_node_->op().nd_sbp_signature()) == sbp_node->sbp_sig_list_[sbp_id]) {
          sbp_node->final_sbp_sig_id_ = sbp_id;
          break;
        }
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitComputationCost(const OpGraph& op_graph) {
  // Compute computation cost for sbp nodes
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    // get corresponding sbp node producer
    SbpNode* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    // get parallel description. Number of devices.
    const ParallelDesc& parallel_desc = op_node->parallel_desc();

    CHECK_EQ_OR_RETURN(sbp_node->cost_.size(), sbp_node->sbp_sig_list_.size());
    auto LogicalBlobDesc4Bn = [&](const std::string& bn) -> const BlobDesc& {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      return op_node->LogicalBlobDesc4Lbi(lbi);
    };
    for (int32_t sbp_id = 0; sbp_id < sbp_node->sbp_sig_list_.size(); sbp_id++) {
      double comp_cost = JUST(op_node->op().GetComputeComplexity(
          &sbp_node->sbp_sig_list_[sbp_id], LogicalBlobDesc4Bn, parallel_desc));
      if (comp_cost > GetValidMaxCopyCost()) {
        sbp_node->cost_[sbp_id] = comp_cost;
      } else {
        sbp_node->cost_[sbp_id] =
            cost_ratio_ * comp_cost
            * JUST(op_node->op().GetInputOutputFastestTimeShape())->elem_cnt();
      }
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

// Init copy cost and memory for edges
Maybe<void> SbpConstructor::InitCopyAndMemoryCost(const OpGraph& op_graph) {
  bool nccl_not_use_compute_stream = !nccl_use_compute_stream_;
  // Compute copy cost for sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // get corresponding sbp node consumer
    SbpNode* sbp_node_consumer = op_name2sbp_node_[op_node->op().op_name()];
    // Initialize copy cost between two nodes
    for (auto* sbp_edge : sbp_node_consumer->edges_in_) {
      // producer sbp node
      const auto* sbp_node_producer = sbp_edge->start_node_;
      // skip it if proxy
      if (!sbp_node_producer->op_node_) { continue; }
      sbp_edge->cost_.resize(sbp_node_producer->sbp_sig_list_.size());
      if (nccl_not_use_compute_stream) {
        sbp_edge->memory_.resize(sbp_node_producer->sbp_sig_list_.size());
      }
      int32_t consumer_sbp_size = sbp_node_consumer->sbp_sig_list_.size();
      // look through sbp signature in producer
      for (int32_t i = 0; i < sbp_node_producer->sbp_sig_list_.size(); ++i) {
        sbp_edge->cost_[i].resize(consumer_sbp_size, 0);
        if (nccl_not_use_compute_stream) { sbp_edge->memory_[i].resize(consumer_sbp_size, 0); }
      }
    }
    // Find all those cases with wait time
    // Do not skip edges carrying no lbi
    sbp_node_consumer->InitCopyAndMemoryCost(use_sbp_collector_, nccl_not_use_compute_stream);
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::ApplyTrunkAlgo() {
  // TODO: Remove this
  auto OpNode2MutableOpCtrlDeps = JUST(GetMutableOpCtrlDeps(*op_graph_));
  // Compute layer number for each node
  int32_t max_min_layer = sbp_graph_.ComputeLayer(op_name2sbp_node_, *OpNode2MutableOpCtrlDeps);
  // Accumulate cost on the trunk after initializing computation cost
  sbp_graph_.FindTrunk(max_min_layer, op_name2sbp_node_);
  return Maybe<void>::Ok();
}

// Load logical blob ids onto sbp edges
void SbpConstructor::LoadLbi2SbpEdge(const OpGraph& op_graph) {
  // Load logical blobs onto sbp edges

  for (auto* sbp_node_consumer : sbp_graph_.node_list_) {
    auto* op_node = sbp_node_consumer->op_node_;

    // Loading logical blobs between two nodes
    // look through input blobs
    for (const std::string& ibn : op_node->op().input_bns()) {
      // Each input blob has one source op node.
      OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
      // producer sbp node
      const auto* sbp_node_producer = op_name2sbp_node_[producer->op().op_name()];
      // TODO: recode this
      auto* edge_found = sbp_node_consumer->FindEdgeWithNode(sbp_node_producer);

      CHECK(edge_found != NULL) << "SbpEdge not found while loading!" << std::endl;

      // Add copy cost for each blob
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      edge_found->LoadLbi(lbi);
    }
  };
}

Maybe<void> SbpConstructor::CheckSbpAgreement(const Job& job) {
  Job new_job;
  new_job.CopyFrom(job);
  OpGraph op_graph(new_job);
  // Compare sbp in job
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    const std::string& op_name = op_node->op().op_name();
    const NdSbpSignature& auto_parallel_sbp =
        NdSbpSignature(job.job_parallel_view_conf().op_name2nd_sbp_signature_conf().at(op_name));
    const NdSbpSignature& new_sbp = op_node->nd_sbp_signature();
    CHECK_EQ_OR_RETURN(auto_parallel_sbp.bn_in_op2nd_sbp_size(), new_sbp.bn_in_op2nd_sbp_size());
    for (const auto& iter : auto_parallel_sbp.bn_in_op2nd_sbp()) {
      const NdSbp& new_sbp_parallel = new_sbp.bn_in_op2nd_sbp().at(iter.first);
      const NdSbp& auto_parallel_sbp = iter.second;
      // According error message, we can find op_type in op_conf.proto with type_id and locate
      // the error op type.
      const std::string& error_mgs =
          "Op: `" + op_name + "`(type_id: " + std::to_string(op_node->op().op_conf().op_type_case())
          + ") changed sbp from " + NdSbpToString(auto_parallel_sbp) + "(AutoParallel) to "
          + NdSbpToString(new_sbp_parallel) + "(OpGraph) with blob_name: `" + iter.first + "`.";
      CHECK_OR_RETURN(new_sbp_parallel == auto_parallel_sbp) << error_mgs;
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

// TODO: delete this, this is for variable op only
Maybe<HashMap<const OpNode*, HashSet<std::string>>> SbpConstructor::GetMutableOpCtrlDeps(
    const OpGraph& op_graph) {
  auto IsMutableConsumedLbi = [](const Operator& op, const LogicalBlobId& lbi) -> bool {
    for (const std::string& bn : op.input_bns()) {
      if (op.BnInOp2Lbi(bn) == lbi && op.InputBlobModifier4Ibn(bn).is_mutable()) { return true; }
    }
    return false;
  };
  const auto& IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  HashMap<const OpNode*, HashSet<std::string>> op_node2ctrl_in_op_names;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
    if (op_node->out_edges().size() <= 1) { return Maybe<void>::Ok(); }
    const Operator& variable_op = op_node->op();
    const LogicalBlobId& variable_lbi = variable_op.BnInOp2Lbi(variable_op.SoleObn());
    const OpNode* mutable_consumer = nullptr;
    std::vector<const OperatorConf*> naive_consumers;
    naive_consumers.reserve(op_node->out_edges().size());
    for (OpEdge* edge : op_node->out_edges()) {
      const auto& op_conf = edge->dst_node()->op().op_conf();
      if (IsMutableConsumedLbi(edge->dst_node()->op(), variable_lbi)) {
        CHECK_OR_RETURN(mutable_consumer == nullptr);
        mutable_consumer = edge->dst_node();
      } else {
        naive_consumers.emplace_back(&op_conf);
      }
    }
    if (mutable_consumer == nullptr) { return Maybe<void>::Ok(); }
    for (const auto* fw_bw_op : naive_consumers) {
      op_node2ctrl_in_op_names[mutable_consumer].insert(fw_bw_op->name());
    }
    return Maybe<void>::Ok();
  }));
  // Filter ctrl edges if all ctrl_in_op_names are reachable
  HashMap<const OpNode*, HashSet<std::string>> filter_op_ctrl_deps;
  for (const auto& pair : op_node2ctrl_in_op_names) {
    const OpNode* op_node = pair.first;
    for (const auto& fw_bw_op_name : pair.second) {
      if (!IsReachable(fw_bw_op_name, op_node->op().op_name())) {
        filter_op_ctrl_deps[op_node].insert(fw_bw_op_name);
      }
    }
  }
  return filter_op_ctrl_deps;
}

void SbpConstructor::InitAvailableMemory() {
  size_t free = 0;
  size_t total = 0;
#ifdef WITH_CUDA
  CudaCurrentDeviceGuard guard(GlobalProcessCtx::Rank());
  OF_CUDA_CHECK(cudaMemGetInfo(&free, &total));
#else
  free = 1e13;   // 10T = 10,000G
  total = 1e13;  // 10T = 10,000G
  LOG(INFO) << "We do not use CUDA in CPU mode, auto memory is unnecessary since all the SBPs are "
               "Broadcast.";
#endif
  // The estimated memory differs from the lower bound of the peak memory by the first ratio.
  // The first ratio varies from -3% to 3.2% if not enabling nccl_use_compute_stream.
  // It varies from 0.00313% to 0.5% if enabling nccl_use_compute_stream.
  double first_ratio = 1.0;
  if (nccl_use_compute_stream_) {
    first_ratio = 1.01;
  } else {
    first_ratio = 1.04;
  }
  // The lower bound of the peak memory differs from the allocated memory by the second ratio.
  // The second ratio varies from 0 to 2.65% if not using pipeline parallelism.
  // It varies from 0 to 5.23% if using pipeline parallelism.
  double second_ratio = 1.06;
  // The occupied memory at this moment would be around 1114MB to 1240MB.
  // When it gets to the training process, the occupied memory might drop by 162MB.
  // But the key is that we start to allocate memory before the training process.
  // Thus, this 161MB should not be added to the free memory.
  // We still use "available memory = free / ratio" instead of "free / ratio + 161MB".
  available_memory_ = int64_t(free / (first_ratio * second_ratio));
  LOG(INFO) << "Free memory: " << free << ", total memory: " << total
            << ", available memory: " << available_memory_;
}

void SbpConstructor::InitWeightedCost() {
  for (auto& sbp_node : sbp_graph_.node_list_) {
    sbp_node->ComputeWeightedCost();
    for (auto& sbp_edge : sbp_node->edges_in_) { sbp_edge->ComputeWeightedCost(); }
  }
}

// Print the graph with SBP in order
void SbpConstructor::PrintSBPGraphDebugInfo() {
  // sbp constructor information
  std::cout << "cost_ratio_:" << cost_ratio_ << std::endl;
  std::cout << "wait_time_:" << sbp_graph_.wait_time_ << std::endl;
  std::cout << "use_sbp_collector_" << use_sbp_collector_ << std::endl;
  std::cout << "Total auto parallel guessed memory: " << sbp_graph_.GetMemory() << std::endl;
  std::cout << "Final memory ratio: " << kMemoryRatio << std::endl;
  // test debug
  std::cout << "Get Into Print Op Graph" << std::endl;
  // Collect op_node
  std::vector<OpNode*> node_list;
  for (const auto& op_name_sbp_node : op_name2sbp_node_) {
    auto* op_node_ = op_name_sbp_node.second->op_node_;
    if (op_node_) { node_list.push_back(op_node_); }
  }

  // test debug
  std::cout << "Deciding order" << std::endl;
  // Decide the order to visit the op
  std::vector<int32_t> order;
  auto_parallel::DecideOrder(node_list, order, [&](OpNode* a, OpNode* b) {
    return a->op().op_name().compare(b->op().op_name()) > 0;
  });
  std::vector<int32_t> str_order;

  // test debug
  std::cout << "Finish deciding order" << std::endl;

  for (int32_t i = 0; i < node_list.size(); i++) {
    OpNode* op_node = node_list[order[i]];
    std::cout << op_node->op().op_name() << " (^_^):" << std::endl;
    // get corresponding sbp node
    const auto& it = op_name2sbp_node_.find(op_node->op().op_name());
    // Print debug information for sbp graph
    CHECK(it != op_name2sbp_node_.end());
    const SbpNode* sbp_node = it->second;
    std::cout << "Computation Cost: " << sbp_node->weighted_cost_[sbp_node->final_sbp_sig_id_];
    std::cout << ", Min Layer: " << sbp_node->min_layer_ << ", Max Layer: " << sbp_node->max_layer_
              << ", Tributary Layer: " << sbp_node->tributary_layer_
              << ", in trunk: " << sbp_node->on_trunk_
              << ", Remain Cost: " << sbp_node->acc_trunk_cost_ << std::endl;
    // Sort before printing
    const auto& op_input_bns = op_node->op().input_bns();
    auto CompareString = [](const std::string& a, const std::string& b) {
      return a.compare(b) > 0;
    };
    auto_parallel::DecideOrder(op_input_bns, str_order, CompareString);
    const NdSbpSignature& sbp_signature = sbp_node->FinalSbpSignature();
    // Print out SBP information for input operator
    for (int32_t j : str_order) {
      const auto& ibn = op_input_bns[j];
      const auto& producer_node = op_node->SrcNode4Ibn(ibn);
      std::cout << "Pre Op:" << producer_node.op().op_name() << ": " << ibn;
      const auto& this_sbp_parallel = sbp_signature.bn_in_op2nd_sbp().at(ibn);
      std::cout << ", " << NdSbpToString(this_sbp_parallel);
      if (RequireSameSbp(op_node, ibn)) { std::cout << ", require same SBP"; }
      std::cout << ", " << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn)).shape();
      std::cout << std::endl;
    }
    // Sort before printing
    const auto& op_output_bns = op_node->op().output_bns();
    auto_parallel::DecideOrder(op_output_bns, str_order, CompareString);
    // Print out SBP information for output blobs
    for (int32_t j : str_order) {
      const auto& obn = op_output_bns[j];
      std::cout << "Out Op:" << obn;
      const auto& this_sbp_parallel = sbp_signature.bn_in_op2nd_sbp().at(obn);
      std::cout << ", " << NdSbpToString(this_sbp_parallel);
      std::cout << ", " << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(obn)).shape();
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

}  // namespace auto_parallel
}  // namespace oneflow
