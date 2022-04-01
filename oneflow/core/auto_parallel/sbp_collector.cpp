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

#include "sbp_collector.h"
#include <string>
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "sbp_constructor.h"

namespace oneflow {

namespace auto_parallel {

// Default constructor for SbpCollector
// Don't allow any special case for broadcast!
SbpCollector::SbpCollector() {
  // initialize Sbp Parallel Universe with broadcast.
  // NdSbp sbp_broadcast;
  // sbp_broadcast.mutable_broadcast_parallel();
  // SbpParallelUniverse[sbp_broadcast] = 0;
  // id2SbpParallel.push_back(sbp_broadcast);
}

// Collect all the possible Sbp Parallel from a NdSbpSignature
void SbpCollector::CollectUniverse(NdSbpSignature& sbp_) {
  auto& bn_in_op2sbp_parallels = *sbp_.mutable_bn_in_op2nd_sbp();
  for (auto& OpSbpPair : bn_in_op2sbp_parallels) {
    if (SbpParallelUniverse.find(OpSbpPair.second) == SbpParallelUniverse.end()) {
      int32_t curr_size = SbpParallelUniverse.size();
      SbpParallelUniverse[OpSbpPair.second] = curr_size;
      id2SbpParallel.push_back(OpSbpPair.second);
    }
  }
}
// Collect all the possible Sbp Parallel from a SbpNode
void SbpCollector::CollectUniverse(SbpNode<NdSbpSignature>* sbp_node) {
  for (auto& sbp_ : sbp_node->SbpSignatureObjList) { CollectUniverse(sbp_); }
}
// Collect all the possible Sbp Parallel from a SbpGraph
void SbpCollector::CollectUniverse(SbpGraph<NdSbpSignature>& sbp_graph) {
  for (auto* sbp_node : sbp_graph.NodeList) { CollectUniverse(sbp_node); }
  accumulator.resize(SbpParallelUniverse.size(), 0);
  bs_buffer.Initialize(SbpParallelUniverse.size());
}
// Initialize sbp proxy with given parallel candidates of a blob
SbpNode<NdSbpSignature>* SbpCollector::InitializePorxy(
    SbpGraph<NdSbpSignature>& sbp_graph,
    std::unordered_set<BinarySet, BinarySetHasher>& ParallelCandidates) {
  // Initialize sbp proxy
  SbpNode<NdSbpSignature>* sbp_proxy = sbp_graph.GenerateNode();
  // move parallel candidates
  for (const BinarySet& parallel_candidate : ParallelCandidates) {
    sbp_proxy->ParallelCandidates.emplace_back(parallel_candidate);
  }
  // Initialize computation cost
  sbp_proxy->Cost.resize(sbp_proxy->ParallelCandidates.size(), 0);

  return sbp_proxy;
}

// TODO: Auto Placement!
// It only collect the same sbp with the same parallel description
// In this moment their hierarchy is the same!

// Initialize copy cost from producer to proxy of producer
void SbpCollector::InitializeCopyCostFromNode2Proxy(SbpNode<NdSbpSignature>* sbp_proxy,
                                                    const LogicalBlobId& lbi) {
  // the only edge from producer  to proxy of producer
  SbpEdge<NdSbpSignature>* sbp_edge = sbp_proxy->EdgesIn[0];
  SbpNode<NdSbpSignature>* sbp_node_producer = sbp_edge->StartNode;
  sbp_edge->Cost.resize(sbp_node_producer->SbpSignatureList.size());
  int32_t consumer_sbp_size = sbp_proxy->ParallelCandidates.size();
  // look through sbp signature in producer
  for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_node_producer->SbpSignatureList.size();
       sbp_id_producer++) {
    sbp_edge->Cost[sbp_id_producer].resize(consumer_sbp_size, 0);
  }

  // Assemble copy cost from producer to proxy of producer
  OpNode* producer = sbp_node_producer->op_node;

  // get parallel description. Number of devices.
  const ParallelDesc& producer_parallel_desc = producer->parallel_desc();
  // Need to be careful, the logical blob description should be independent to current
  // NdSbp. Use producer or op_node?
  const BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
  const std::string& obn = *CHECK_JUST(producer->op().obn4lbi(lbi));

  // A buffer to store the sbp parallel id
  std::vector<int32_t> sbp_parallel_ids;

  // look through sbp signature in producer
  for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_node_producer->SbpSignatureList.size();
       sbp_id_producer++) {
    // get sbp parallel for a logical blob in producer
    const auto producer_sbp_bn_in_op2sbp_parallel =
        sbp_node_producer->SbpSignatureList[sbp_id_producer]->bn_in_op2nd_sbp();
    const NdSbp& sbp_producer = producer_sbp_bn_in_op2sbp_parallel.at(obn);

    // look through sbp parallel set in consumer
    for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
      BinarySet& sbp_parallel_set = sbp_proxy->ParallelCandidates[sbp_id_consumer];
      sbp_parallel_set.QuickOutPut(sbp_parallel_ids);

      // look through all sbp parallels in a sbp parallel set
      for (int32_t sbp_parallel_id : sbp_parallel_ids) {
        // get sbp parallel for a logical blob in consumer
        const NdSbp& sbp_consumer = id2SbpParallel[sbp_parallel_id];

        // compute copy cost for a specific logical blob
        // Use the parallel description of producer as those for consumer for now.
        sbp_edge->Cost[sbp_id_producer][sbp_id_consumer] +=
            CHECK_JUST(ComputeCopyCostWithMiddleNodes(sbp_producer, sbp_consumer, logical_blob_desc,
                                                      producer_parallel_desc,
                                                      producer_parallel_desc, /*is_same=*/false));
      }
    }
  }
}

// Initialize copy cost from proxy of producer to consumers
void SbpCollector::InitializeCopyCostFromProxy2Consumer(
    SbpNode<NdSbpSignature>* sbp_proxy,
    const std::vector<std::pair<const OpNode*, std::string>>& consumer_bns,
    HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node) {
  // Connect sbp proxy and consumers
  for (const auto& consumer_bn : consumer_bns) {
    // consumer in cost model
    SbpNode<NdSbpSignature>* sbp_node_consumer =
        op_name2sbp_node[consumer_bn.first->op().op_name()];
    // input blob name of logical blob in consumer
    const std::string& ibn = consumer_bn.second;

    // check is_mutable in consumer
    OpNode* consumer = sbp_node_consumer->op_node;
    CHECK(!IsSameSbp(consumer, ibn)) << "Create a proxy for an unsuitable consumer!\n";

    // Connect sbp proxy and consumer
    sbp_proxy->PointTo(sbp_node_consumer);
    // the sbp edge connecting proxy and consumer
    SbpEdge<NdSbpSignature>* sbp_edge = FindEdgeBetweenNodes(sbp_proxy, sbp_node_consumer);
    sbp_edge->Cost.resize(sbp_proxy->ParallelCandidates.size());
    int32_t consumer_sbp_size = sbp_node_consumer->SbpSignatureList.size();

    // look through sbp parallel set in proxy
    for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_proxy->ParallelCandidates.size();
         sbp_id_producer++) {
      // initialization for copy cost
      sbp_edge->Cost[sbp_id_producer].resize(consumer_sbp_size, 0);
      // get sbp parallel set for a logical blob in proxy
      BinarySet& parallel_candidate = sbp_proxy->ParallelCandidates[sbp_id_producer];

      // look through sbp signatures in consumers
      for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
        // get sbp parallel for a logical blob in consumer
        const auto consumer_sbp_bn_in_op2sbp_parallel =
            sbp_node_consumer->SbpSignatureList[sbp_id_consumer]->bn_in_op2nd_sbp();
        const NdSbp& sbp_consumer = consumer_sbp_bn_in_op2sbp_parallel.at(ibn);

        if ((!parallel_candidate.CheckExistency(SbpParallelUniverse[sbp_consumer]))) {
          sbp_edge->Cost[sbp_id_producer][sbp_id_consumer] = GetMaxVal<float>();
        }
      }
    }
  }
}

// Export list of possible combination of Sbp Parallels
void SbpCollector::ProxySbpCandidate(
    const OpGraph& op_graph, HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node,
    SbpGraph<NdSbpSignature>& sbp_graph) {
  // If needed, we can output the mapping from operator name to its proxy.
  // HashMap<std::string, HashMap<LogicalBlobId, SbpNode<NdSbpSignature>*>>&
  //     op_name2lbi2sbp_proxy;

  // mapping from a logical blob id to a group of consumers and corresponding input blob names.
  // mapping from consumers and input blob names to an unordered_set of SBP Parallel.
  HashMap<std::pair<std::string, LogicalBlobId>,
          HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>>
      producer_lbi2consumer_bn2sbp_set;
  // mapping from a logical blob id to index
  HashMap<LogicalBlobId, int32_t> lbi2index;
  // mapping from the index to producer, consuemr and corresponding input blob name, possible sbp
  // sets
  std::vector<const OpNode*> index2producer;
  std::vector<std::vector<std::pair<const OpNode*, std::string>>> index2consumer_bns;
  std::vector<std::unordered_set<int32_t>> index2sbp_set;

  for (auto* consumer_sbp_node : sbp_graph.NodeList) {
    auto* node = consumer_sbp_node->op_node;

    OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
    // If not support boxing, just skip it.
    if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
    for (const std::string& ibn : node->op().input_bns()) {
      // Skip those blobs who enforc same SBP.
      if (IsSameSbp(node, ibn)) {
        // Enforcing same SBP. Can not collect sbp from this blob.
        continue;
      }

      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode& producer = node->ProducerOpNode4Lbi(lbi);

      // not building proxy for fixed opertors
      if (op_name2sbp_node.find(producer.op().op_name()) == op_name2sbp_node.end()) { return; }
      // decide the index of a logical blob description
      const auto& iterator_lbi = lbi2index.find(lbi);
      int32_t index = 0;
      if (iterator_lbi == lbi2index.end()) {
        index = lbi2index.size();
        lbi2index[lbi] = index;
        // map from lbi to the producer
        index2producer.push_back(&producer);
        // Initialize consumer_bns and the sbp sets
        index2consumer_bns.resize(index + 1);
        index2sbp_set.resize(index + 1);
      } else {
        index = iterator_lbi->second;
      }
      // Add the consumer and corresponding input blob name
      index2consumer_bns[index].push_back({consumer_sbp_node->op_node, ibn});

      // a set to store the id of all possible SBP Parallel for a downstream op
      // should filter out B and other repeated SBP Parallel by pre-storing them into an
      // unordered_set
      std::unordered_set<int32_t>& SbpParallelIDs = index2sbp_set[index];
      // TODO: use SbpSignatureList instead of SbpSignatureObjList
      for (auto& sbp_sig : consumer_sbp_node->SbpSignatureObjList) {
        const auto& map = sbp_sig.bn_in_op2nd_sbp();
        const auto& iter = map.find(ibn);
        CHECK(iter != map.end()) << "blob_name " << ibn << " not found in sbp signature";
        const NdSbp& consumer_sbp = iter->second;
        // filter out repeated SBP
        SbpParallelIDs.insert(SbpParallelUniverse[consumer_sbp]);
      }
    }
  };

  // A set of binary set with broadcast only
  // std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidatesInitializer;
  // BinarySet one_broadcast(SbpParallelUniverse.size());
  // one_broadcast.AddEntry(0);
  // ParallelCandidatesInitializer.insert(std::move(one_broadcast));

  // Decide if we should insert a proxy for each logical blob
  for (auto& lbi_index : lbi2index) {
    int32_t index = lbi_index.second;
    // Only insert proxy for those blobs with multiple downstream consumers.
    if (index2consumer_bns[index].size() < 2) { continue; }
    // Maximum number of possible sbp in the proxy
    int32_t max_num_sbp_proxy = std::min(max_num_sbp_proxy_, index2consumer_bns[index].size());
    // producer in cost model
    const std::string& producer_name = index2producer[index]->op().op_name();
    SbpNode<NdSbpSignature>* sbp_node_producer = op_name2sbp_node[producer_name];

    const LogicalBlobId& lbi = lbi_index.first;
    // store all the binary sets of SBP Parallel into an unordered_set.
    std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidates;

    DfsSbpSet(0, max_num_sbp_proxy, index2sbp_set[index], index2sbp_set[index].begin(),
              ParallelCandidates);

    // Initialize sbp proxy
    SbpNode<NdSbpSignature>* sbp_proxy = InitializePorxy(sbp_graph, ParallelCandidates);
    // Might be unnecessary
    // op_name2lbi2sbp_proxy[producer_name][lbi] = sbp_proxy;

    // Transfer a logical blob from producer to a sbp proxy of this blob
    sbp_node_producer->PointTo(sbp_proxy);

    // Compute copy cost between producer and proxy
    InitializeCopyCostFromNode2Proxy(sbp_proxy, lbi);

    // Build connection and compute copy cost between proxy and consumers
    InitializeCopyCostFromProxy2Consumer(sbp_proxy, index2consumer_bns[index], op_name2sbp_node);

    // Unloading
    for (const auto& consumer_bn : index2consumer_bns[index]) {
      // consumer in cost model
      SbpNode<NdSbpSignature>* sbp_node_consumer =
          op_name2sbp_node[consumer_bn.first->op().op_name()];
      // the sbp edge connecting producer and consumer
      SbpEdge<NdSbpSignature>* edge_found =
          FindEdgeBetweenNodes(sbp_node_producer, sbp_node_consumer);
      // unload logical blob from sbp edges
      edge_found->UnloadLbi(lbi);
      // Do not clip this edge. Save it for wait time.
      // clip this edge if it no longer carrys any blob
      // We don't clip edges now since we have transfer cost
      // if (edge_found->EmptyLbi() && edge_found->WaitTime <= 0.0 && edge_found->WaitTime > -0.5)
      // sbp_graph.ClipEdge(edge_found);
    }
  }
}

// Depth first search to collect Sbp Parallel information for different lbis
void SbpCollector::DfsSbpSet(int32_t depth, int32_t max_depth,
                             const std::unordered_set<int32_t>& sbp_sets,
                             const std::unordered_set<int32_t>::iterator start_it,
                             std::unordered_set<BinarySet, BinarySetHasher>& ParallelCandidates) {
  if (depth > 0) {
    // store the binary set into an unordered_set
    ParallelCandidates.insert(bs_buffer);
  }
  if (depth >= max_depth) { return; }

  // go through the rest of the sbp parallel
  std::unordered_set<int32_t>::iterator curr_it = start_it;
  while (curr_it != sbp_sets.end()) {
    // Take the value out
    int32_t SbpParallelNum = *curr_it;
    // Then move to the next pointer
    ++curr_it;
    if (accumulator[SbpParallelNum] == 0) {
      bs_buffer.AddEntry(SbpParallelNum);
      ++accumulator[SbpParallelNum];
      DfsSbpSet(depth + 1, max_depth, sbp_sets, curr_it, ParallelCandidates);
      bs_buffer.DeleteEntry(SbpParallelNum);
      --accumulator[SbpParallelNum];
    }
  }
}

}  // namespace auto_parallel

}  // namespace oneflow
