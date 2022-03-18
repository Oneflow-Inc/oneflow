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

#include <string>
#include "sbp_collector.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
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
    HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>& consumer_bn2sbp_set,
    HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node) {
  // Connect sbp proxy and consumers
  for (const auto& consumer_bn_group : consumer_bn2sbp_set) {
    // consumer in cost model
    SbpNode<NdSbpSignature>* sbp_node_consumer = op_name2sbp_node[consumer_bn_group.first.first];
    // input blob name of logical blob in consumer
    const std::string& ibn = consumer_bn_group.first.second;

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
      // a set to store the id of all possible SBP Parallel for a downstream op
      // should filter out B and other repeated SBP Parallel by pre-storing them into an
      // unordered_set
      std::unordered_set<int32_t>& SbpParallelIDs = producer_lbi2consumer_bn2sbp_set[{
          producer.op().op_name(), lbi}][{node->op().op_name(), ibn}];
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
  std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidatesInitializer;
  // BinarySet one_broadcast(SbpParallelUniverse.size());
  // one_broadcast.AddEntry(0);
  // ParallelCandidatesInitializer.insert(std::move(one_broadcast));

  // Decide if we should insert a proxy for each logical blob
  for (auto& lbi7groups : producer_lbi2consumer_bn2sbp_set) {
    // Only insert proxy for those blobs with multiple downstream consumers.
    if (lbi7groups.second.size() < 2) { continue; }
    const std::string& producer_name = lbi7groups.first.first;
    // producer in cost model
    SbpNode<NdSbpSignature>* sbp_node_producer = op_name2sbp_node[producer_name];
    const LogicalBlobId& lbi = lbi7groups.first.second;
    HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>& consumer_bn2sbp_set =
        lbi7groups.second;
    HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it_begin =
        consumer_bn2sbp_set.begin();
    // store all the binary sets of SBP Parallel into an unordered_set.
    std::unordered_set<BinarySet, BinarySetHasher> ParallelCandidates(
        ParallelCandidatesInitializer);

    DfsSbpSet(it_begin, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
    // Initialize sbp proxy
    SbpNode<NdSbpSignature>* sbp_proxy = InitializePorxy(sbp_graph, ParallelCandidates);
    // Might be unnecessary
    // op_name2lbi2sbp_proxy[producer_name][lbi] = sbp_proxy;

    // Transfer a logical blob from producer to a sbp proxy of this blob
    sbp_node_producer->PointTo(sbp_proxy);

    // Compute copy cost between producer and proxy
    InitializeCopyCostFromNode2Proxy(sbp_proxy, lbi);

    // Build connection and compute copy cost between proxy and consumers
    InitializeCopyCostFromProxy2Consumer(sbp_proxy, consumer_bn2sbp_set, op_name2sbp_node);

    // Unloading
    for (const auto& consumer_bn_group : consumer_bn2sbp_set) {
      // consumer in cost model
      SbpNode<NdSbpSignature>* sbp_node_consumer = op_name2sbp_node[consumer_bn_group.first.first];
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
void SbpCollector::DfsSbpSet(
    HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it,
    HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>& consumer_bn2sbp_set,
    HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node,
    std::unordered_set<BinarySet, BinarySetHasher>& ParallelCandidates) {
  if (it == consumer_bn2sbp_set.end()) {
    // store the binary set into an unordered_set
    ParallelCandidates.insert(bs_buffer);
  } else {
    const std::string& consumer_name = it->first.first;
    const std::string& ibn = it->first.second;
    SbpNode<NdSbpSignature>* consumer_sbp_node = op_name2sbp_node[consumer_name];
    // a set to store the id of all possible SBP Parallel for a downstream op
    // should filter out B and other repeated SBP Parallel by pre-storing them into an
    // unordered_set
    std::unordered_set<int32_t> SbpParallelIDs;
    for (auto& sbp_sig : consumer_sbp_node->SbpSignatureObjList) {
      const auto& map = sbp_sig.bn_in_op2nd_sbp();
      const auto& iter = map.find(ibn);
      CHECK(iter != map.end()) << "blob_name " << ibn << " not found in sbp signature";
      const NdSbp& consumer_sbp = iter->second;
      SbpParallelIDs.insert(SbpParallelUniverse[consumer_sbp]);
    }
    // next iterator
    HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it_next =
        it;
    ++it_next;
    // go through all the sbp parallel of different candidates
    for (int32_t SbpParallelNum : SbpParallelIDs) {
      if (++accumulator[SbpParallelNum] == 1) {
        bs_buffer.AddEntry(SbpParallelNum);
        DfsSbpSet(it_next, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
        bs_buffer.DeleteEntry(SbpParallelNum);
      } else {
        DfsSbpSet(it_next, consumer_bn2sbp_set, op_name2sbp_node, ParallelCandidates);
      }
      accumulator[SbpParallelNum]--;
    }
  }
}

}  // namespace auto_parallel

}  // namespace oneflow
