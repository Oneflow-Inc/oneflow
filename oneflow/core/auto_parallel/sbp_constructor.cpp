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
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "sbp_collector.h"

namespace oneflow {

namespace auto_parallel {

Maybe<void> SbpConstructor::Init(const OpGraph& op_graph, Job* job /*Maybe not use*/) {
  JUST(InitSbpGraph(op_graph, *job));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitSbpGraph(const OpGraph& op_graph, const Job& job) {
  // TODO: process mirrored node
  JUST(GenerateNodeAndEdge(op_graph, job));
  JUST(FillSbpSignatureForOpNode(op_graph, job));
  JUST(InitComputationCost(op_graph));
  if (enable_mainstem_algo_) { JUST(ApplyMainstemAlgo()); }
  if (use_sbp_collector_) {
    // Load logical blobs on all sbp edges.
    LoadLbi2SbpEdge(op_graph);
    // Use sbp collector to create sbp proxy for nodes with multiple downstream operators.
    SbpCollector sbp_collector;
    sbp_collector.CollectUniverse(sbp_graph_);
    sbp_collector.ProxySbpCandidate(op_graph, op_name2sbp_node_, sbp_graph_);
  }
  JUST(InitCopyCost(op_graph));
  sbp_graph_.RandomSbpSignature(use_sbp_collector_);
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::FindBestSbpSignature() {
  double ori_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Initial cost: " << ori_cost;
  int elimination_num = sbp_graph_.NodeAndEdgeEliminations();
  LOG(INFO) << "Elimination number: " << elimination_num;
  sbp_graph_.GreedyStrategy(5);
  sbp_graph_.FinalizeSbp();

  double final_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Final cost: " << final_cost;
  if (ori_cost + 1.0 < final_cost) { LOG(WARNING) << "ori_cost less than final_cost!!!"; }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::DumpNdSbpSignatureForJob(const OpGraph& op_graph, Job* job) {
  op_graph.ForEachNode([&](const OpNode* node) -> void {
    SbpNode<cfg::SbpSignature>* sbp_node = op_name2sbp_node_[node->op().op_name()];
    // Update NdSbpSignature
    cfg::NdSbpSignature nd_sbp_signature;
    SbpSignatureToNdSbpSignature(*sbp_node->FinalSbpSignature(), &nd_sbp_signature);
    nd_sbp_signature.ToProto(
        &(*job->mutable_job_parallel_view_conf()
               ->mutable_op_name2nd_sbp_signature_conf())[node->op().op_name()]);
    // Update SbpSignature
    sbp_node->FinalSbpSignature()->ToProto(
        &(*job->mutable_job_parallel_view_conf()
               ->mutable_op_name2sbp_signature_conf())[node->op().op_name()]);
    // TODO: Specially update sbp conf by using polymorphism function
    // Update sbp for variable op
    if (node->op().op_conf().has_variable_conf()) {
      for (auto& op : *job->mutable_net()->mutable_op()) {
        if (op.name() == node->op().op_name()) {
          op.mutable_variable_conf()->mutable_nd_sbp()->Clear();
          op.mutable_variable_conf()->mutable_nd_sbp()->Add(
              SbpParallelToString(sbp_node->FinalSbpSignature()->bn_in_op2sbp_parallel()["out"]));
        }
      }
    }
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::GenerateNodeAndEdge(const OpGraph& op_graph, const Job& job) {
  JobParallelViewConf job_parallel_view_conf(job.job_parallel_view_conf());
  // Create sbp nodes
  op_graph.ForEachNode([&](OpNode* op_node) {
    // TODO: support mirror op
    bool is_mirrored_conf = false;
    {
      const auto& op_name2is_mirrored = job_parallel_view_conf.op_name2is_mirrored_parallel_view();
      const auto& iter = op_name2is_mirrored.find(op_node->op().op_name());
      if (iter != op_name2is_mirrored.end()) { is_mirrored_conf = iter->second; }
    }
    CHECK(is_mirrored_conf == false) << "Haven't deal with mirror operators.";
    // Generate sbp node in cost model and link it with corresponding op node
    SbpNode<cfg::SbpSignature>* sbp_node = sbp_graph_.GenerateNode();
    // Mapping from sbp_node to op_node
    sbp_node->op_node = op_node;  // TODO: SetOpNode()
    op_name2sbp_node_[op_node->op().op_name()] = sbp_node;
  });
  // Create sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // Get corresponding sbp node
    SbpNode<cfg::SbpSignature>* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    for (const auto op_edge : op_node->out_edges()) {
      const auto& end_node_name = op_edge->dst_node()->op().op_name();
      // Generate sbp edge in cost model
      sbp_node->PointTo(op_name2sbp_node_[end_node_name]);
    }
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::FillSbpSignatureForOpNode(const OpGraph& op_graph, const Job& job) {
  // TODO: use user sbp signature in JobParallelViewConf
  // const JobParallelViewConf& job_parallel_view_conf(job.job_parallel_view_conf());
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    HashMap<std::string, NdSbpInferHint> ibn2nd_sbp_infer_hint;
    for (const std::string& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
      const ParallelDesc* parallel_desc = &producer->parallel_desc();
      const BlobDesc* logical_blob_desc = &producer->LogicalBlobDesc4Lbi(lbi);
      const cfg::NdSbp* nd_sbp = &producer->NdSbp4Lbi(lbi);
      ibn2nd_sbp_infer_hint.emplace(ibn, NdSbpInferHint(parallel_desc, logical_blob_desc, nd_sbp));
    }
    const auto NdSbpInferHint4Ibn = [&](const std::string& bn) -> Maybe<const NdSbpInferHint*> {
      auto it = ibn2nd_sbp_infer_hint.find(bn);
      CHECK_OR_RETURN(it != ibn2nd_sbp_infer_hint.end());
      return Maybe<const NdSbpInferHint*>(&it->second);
    };
    // Get all valid sbp_signatures
    const std::shared_ptr<cfg::SbpSignatureList>& sbp_list =
        JUST(op_node->op().GetValidSbpSignatureList(op_node->parallel_desc(), NdSbpInferHint4Ibn));
    CHECK_GT_OR_RETURN(sbp_list->sbp_signature_size(), 0);
    // Fill in sbp_node
    SbpNode<cfg::SbpSignature>* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    for (int i = 0; i < sbp_list->sbp_signature_size(); ++i) {
      sbp_node->SbpSignatureObjList.emplace_back(sbp_list->sbp_signature(i));
    }
    sbp_node->InitializeSbp();
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitComputationCost(const OpGraph& op_graph) {
  // Compute computation cost for sbp nodes
  op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    // get corresponding sbp node producer
    SbpNode<cfg::SbpSignature>* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    // get parallel description. Number of devices.
    const ParallelDesc& parallel_desc = op_node->parallel_desc();

    CHECK_EQ_OR_RETURN(sbp_node->Cost.size(), sbp_node->SbpSignatureList.size());
    auto logical_blob_desc4bn = [&](const std::string& bn) -> const BlobDesc& {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      return op_node->LogicalBlobDesc4Lbi(lbi);
    };
    for (int32_t sbp_id = 0; sbp_id < sbp_node->SbpSignatureList.size(); sbp_id++) {
      double comp_cost = JUST(op_node->op().GetComputeComplexity(
          sbp_node->SbpSignatureList[sbp_id], logical_blob_desc4bn, parallel_desc));
      if (comp_cost > cut_cost) {
        sbp_node->Cost[sbp_id] = comp_cost;
      } else {
        sbp_node->Cost[sbp_id] = cost_ratio_ * comp_cost;
      }
    }
    return Maybe<void>::Ok();
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitCopyCost(const OpGraph& op_graph) {
  // Compute copy cost for sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // get corresponding sbp node consumer
    SbpNode<cfg::SbpSignature>* sbp_node_consumer = op_name2sbp_node_[op_node->op().op_name()];
    // Initialize copy cost between two nodes
    for (auto* sbp_edge : sbp_node_consumer->EdgesIn) {
      // producer sbp node
      const auto* sbp_node_producer = sbp_edge->StartNode;
      // skip it if proxy
      if (!sbp_node_producer->op_node) { continue; }
      sbp_edge->Cost.resize(sbp_node_producer->SbpSignatureList.size());
      int32_t consumer_sbp_size = sbp_node_consumer->SbpSignatureList.size();
      // look through sbp signature in producer
      for (int32_t i = 0; i < sbp_node_producer->SbpSignatureList.size(); ++i) {
        sbp_edge->Cost[i].resize(consumer_sbp_size, 0);
      }
    }
    // Find all those cases with wait time
    // Do not skip edges carrying no lbi
    sbp_node_consumer->InitializeCopyCost(false, use_sbp_collector_);
    for (auto* sbp_edge : sbp_node_consumer->EdgesIn) {
      // skip it if proxy
      if (!sbp_edge->StartNode->op_node) { continue; }
      // Reset Wait time
      for (int32_t i = 0; i < sbp_edge->Cost.size(); ++i) {
        for (int32_t j = 0; j < sbp_edge->Cost[i].size(); ++j) {
          // If transferring between devices, we need to add wait time.
          if (sbp_edge->Cost[i][j] > 0.0) { sbp_edge->Cost[i][j] = sbp_edge->WaitTime; }
        }
      }
    }

    // Re-compute the costs, skip edges carrying no lbi
    sbp_node_consumer->InitializeCopyCost(true, use_sbp_collector_);
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::ApplyMainstemAlgo() {
  // Compute layer number for each node
  int32_t max_MinLayer = sbp_graph_.ComputeLayer(op_name2sbp_node_);
  // Accumulate cost on the mainstem after initializing computation cost
  sbp_graph_.FindMainstem(max_MinLayer, op_name2sbp_node_);
  return Maybe<void>::Ok();
}

// Load logical blob ids onto sbp edges
void SbpConstructor::LoadLbi2SbpEdge(const OpGraph& op_graph) {
  // Load logical blobs onto sbp edges

  for (auto* sbp_node_consumer : sbp_graph_.NodeList) {
    auto* op_node = sbp_node_consumer->op_node;

    // Loading logical blobs between two nodes
    // look through input blobs
    for (const std::string& ibn : op_node->op().input_bns()) {
      // Each input blob has one source op node.
      OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
      // producer sbp node
      const auto* sbp_node_producer = op_name2sbp_node_[producer->op().op_name()];
      // TODO: recode this
      auto* edge_found = auto_parallel::FindEdgeBetweenNodes(sbp_node_producer, sbp_node_consumer);

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
    const cfg::SbpSignature& auto_parallel_sbp =
        cfg::SbpSignature(job.job_parallel_view_conf().op_name2sbp_signature_conf().at(op_name));
    const cfg::SbpSignature& new_sbp = op_node->sbp_signature();
    CHECK_EQ_OR_RETURN(auto_parallel_sbp.bn_in_op2sbp_parallel_size(),
                       new_sbp.bn_in_op2sbp_parallel_size());
    for (const auto& iter : auto_parallel_sbp.bn_in_op2sbp_parallel()) {
      const cfg::SbpParallel& new_sbp_parallel = new_sbp.bn_in_op2sbp_parallel().at(iter.first);
      const cfg::SbpParallel& auto_parallel_sbp = iter.second;
      // According error message, we can find op_type in op_conf.proto with type_id and locate
      // the error op type.
      const std::string& error_mgs =
          "Op: `" + op_name + "`(type_id: " + std::to_string(op_node->op().op_conf().op_type_case())
          + ") changed sbp from " + SbpParallelToString(auto_parallel_sbp) + "(AutoParallel) to "
          + SbpParallelToString(new_sbp_parallel) + "(OpGraph) with blob_name: `" + iter.first
          + "`.";
      CHECK_OR_RETURN(new_sbp_parallel == auto_parallel_sbp) << error_mgs;
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

// Print the graph with SBP in order
void SbpConstructor::PrintSBPGraphDebugInfo() {
  // sbp constructor information
  std::cout << "cost_ratio_:" << cost_ratio_ << std::endl;
  std::cout << "transfer_cost_:" << sbp_graph_.transfer_cost << std::endl;
  std::cout << "wait_time_:" << sbp_graph_.wait_time << std::endl;
  std::cout << "use_sbp_collector_" << use_sbp_collector_ << std::endl;
  // test debug
  std::cout << "Get Into Print Op Graph" << std::endl;
  // Collect op_node
  std::vector<OpNode*> NodeList;
  for (const auto& op_name_sbp_node : op_name2sbp_node_) {
    auto* op_node_ = op_name_sbp_node.second->op_node;
    if (op_node_) { NodeList.push_back(op_node_); }
  }

  // test debug
  std::cout << "Deciding order" << std::endl;
  // Decide the order to visit the op
  std::vector<int32_t> order;
  auto_parallel::DecideOrder(NodeList, order, [&](OpNode* a, OpNode* b) {
    return a->op().op_name().compare(b->op().op_name()) > 0;
  });
  std::vector<int32_t> str_order;

  // test debug
  std::cout << "Finish deciding order" << std::endl;

  for (int32_t i = 0; i < NodeList.size(); i++) {
    OpNode* op_node = NodeList[order[i]];
    std::cout << op_node->op().op_name() << " (^_^):" << std::endl;
    // get corresponding sbp node
    auto it = op_name2sbp_node_.find(op_node->op().op_name());
    // Print debug information for sbp graph
    CHECK(it != op_name2sbp_node_.end());
    const SbpNode<cfg::SbpSignature>* sbp_node = it->second;
    std::cout << "Computation Cost: " << sbp_node->Cost[sbp_node->FinalSbpSignatureId];
    std::cout << ", Min Layer: " << sbp_node->MinLayer << ", Max Layer: " << sbp_node->MaxLayer
              << ", Tributary Layer: " << sbp_node->TributaryLayer
              << ", in mainstem: " << sbp_node->IfMainstem
              << ", Remain Cost: " << sbp_node->AccMainstemCost << std::endl;
    // Sort before printing
    const auto& op_input_bns = op_node->op().input_bns();
    auto comp = [](const std::string& a, const std::string& b) { return a.compare(b) > 0; };
    auto_parallel::DecideOrder(op_input_bns, str_order, comp);
    const cfg::SbpSignature& sbp_signature = *sbp_node->FinalSbpSignature();
    // Print out SBP information for input operator
    for (int32_t j : str_order) {
      const auto& ibn = op_input_bns[j];
      auto producer_node = op_node->MutSrcNode4Ibn(ibn);
      std::cout << "Pre Op:" << producer_node->op().op_name() << ": " << ibn;
      const auto& this_sbp_parallel = sbp_signature.bn_in_op2sbp_parallel()[ibn];
      std::cout << ", " << SbpParallelToString(this_sbp_parallel);
      const auto input_blob_modifier_ = op_node->op().InputBlobModifier4Ibn(ibn);
      if (IsSameSBP(op_node, ibn)) { std::cout << ", same SBP"; }
      std::cout << ", "
                << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn)).shape().elem_cnt();
      std::cout << std::endl;
    }
    // Sort before printing
    const auto& op_output_bns = op_node->op().output_bns();
    auto_parallel::DecideOrder(op_output_bns, str_order, comp);
    // Print out SBP information for output blobs
    for (int32_t j : str_order) {
      const auto& obn = op_output_bns[j];
      std::cout << "Out Op:" << obn;
      const auto& this_sbp_parallel = sbp_signature.bn_in_op2sbp_parallel()[obn];
      std::cout << ", " << SbpParallelToString(this_sbp_parallel);
      std::cout << ", "
                << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(obn)).shape().elem_cnt();
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

}  // namespace auto_parallel
}  // namespace oneflow
