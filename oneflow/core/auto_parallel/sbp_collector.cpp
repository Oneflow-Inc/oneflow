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
#include "oneflow/core/auto_parallel/sbp_collector.h"
#include "oneflow/core/auto_parallel/binary_set.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/auto_parallel/sbp_constructor.h"

namespace oneflow {

namespace auto_parallel {

namespace {
// Whether the given binary set intersects all the sbp sets of the consumers
bool IfIntersectAll(
    const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
    const BinarySet& bs) {
  for (const auto& sbp_set_group : consumer_bn2sbp_set) {
    if (!bs.IfIntersect(sbp_set_group.second)) { return false; }
  }

  return true;
}

// Find unique sbp sets
void FindUniqueSbpSets(
    const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
    const std::unordered_set<int32_t>& all_sbp_set, std::vector<int32_t>& accumulator,
    BinarySet& unique_sbps) {
  std::vector<int32_t> sbp_ids;
  // count the number of sbp
  for (const auto& sbp_set_group : consumer_bn2sbp_set) {
    sbp_set_group.second.QuickOutput(sbp_ids);
    for (int32_t sbp_id : sbp_ids) { accumulator[sbp_id]++; }
  }
  // find unique sbp and clear the accumulator
  for (const auto& sbp_id : all_sbp_set) {
    if (accumulator[sbp_id] == 1) { unique_sbps.AddEntry(sbp_id); }
    accumulator[sbp_id] = 0;
  }
}

// Find unique sbp groups
void FindUniqueSbpGroups(
    const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
    const std::unordered_set<int32_t>& all_sbp_set, std::vector<int32_t>& accumulator,
    BinarySet& bs_buffer, std::vector<BinarySet>& unique_sbp_groups) {
  // find the unique sbp sets
  BinarySet unique_sbps(accumulator.size());
  FindUniqueSbpSets(consumer_bn2sbp_set, all_sbp_set, accumulator, unique_sbps);

  // A: {B, S0, S1, S2, S3}, C: {B, S0}, D: {B, S0}
  // {S1, S2, S3} show up only once, a parallel candidate should not contain two of them
  for (const auto& sbp_set_group : consumer_bn2sbp_set) {
    unique_sbps.IntersectionTo(sbp_set_group.second, bs_buffer);
    // Find those unique sbp groups with more than two sbp
    // For example {B, S1, S2} is an impossible proxy candidate,
    // since {S1, S2} is only contained by A but not contained by C and D.
    // A could be either S1 or S2. The tensor do not need to be transferred to both S1 and S2.
    if (bs_buffer.Total() >= 2) { unique_sbp_groups.push_back(bs_buffer); }
  }
  bs_buffer.Clear();
}

// If not contains two sbp from a same unique group
bool No2SbpFromSameUniqueGroup(const BinarySet& bs,
                               const std::vector<BinarySet>& unique_sbp_groups) {
  BinarySet intersection(bs.GetSizeOfSet());
  for (const auto& unique_sbp_group : unique_sbp_groups) {
    bs.IntersectionTo(unique_sbp_group, intersection);
    // For example {B, S1, S2} is an impossible proxy candidate,
    // since {S1, S2} is only contained by A but not contained by C and D.
    // A could be either S1 or S2. The tensor do not need to be transferred to both S1 and S2.
    if (intersection.Total() >= 2) { return false; }
  }
  return true;
}
}  // namespace

// Default constructor for SbpCollector
// Don't allow any special case for broadcast!
SbpCollector::SbpCollector() {
  // initialize Sbp Parallel Universe with broadcast.
  // NdSbp sbp_broadcast;
  // sbp_broadcast.mutable_broadcast_parallel();
  // nd_sbp_universe_[sbp_broadcast] = 0;
  // id2nd_sbp_.push_back(sbp_broadcast);
}

// Collect all the possible Sbp Parallel from a NdSbpSignature
void SbpCollector::CollectUniverse(const NdSbpSignature& nd_sbp_sig) {
  for (auto& bn_sbp_pair : nd_sbp_sig.bn_in_op2nd_sbp()) {
    if (nd_sbp_universe_.find(bn_sbp_pair.second) == nd_sbp_universe_.end()) {
      int32_t curr_size = nd_sbp_universe_.size();
      nd_sbp_universe_[bn_sbp_pair.second] = curr_size;
      id2nd_sbp_.push_back(bn_sbp_pair.second);
    }
  }
}
// Collect all the possible Sbp Parallel from a SbpNode
void SbpCollector::CollectUniverse(const SbpNode* sbp_node) {
  for (auto& nd_sbp_sig : sbp_node->sbp_sig_list_) { CollectUniverse(nd_sbp_sig); }
}
// Collect all the possible Sbp Parallel from a SbpGraph
void SbpCollector::CollectUniverse(const SbpGraph& sbp_graph) {
  for (auto* sbp_node : sbp_graph.node_list_) { CollectUniverse(sbp_node); }
  accumulator_.resize(nd_sbp_universe_.size(), 0);
  bs_buffer_.Initialize(nd_sbp_universe_.size());
}

// TODO: Auto Placement!
// It only collect the same sbp with the same parallel description
// In this moment their hierarchy is the same!

// Initialize copy cost from producer to proxy of producer
void SbpCollector::InitializeCopyCostFromNode2Proxy(const SbpNode* sbp_proxy,
                                                    const LogicalBlobId& lbi) const {
  // the only edge from producer  to proxy of producer
  SbpEdge* sbp_edge = sbp_proxy->edges_in_[0];
  SbpNode* sbp_node_producer = sbp_edge->start_node_;
  sbp_edge->cost_.resize(sbp_node_producer->sbp_sig_list_.size());
  int32_t consumer_sbp_size = sbp_proxy->parallel_candidates_.size();
  // look through sbp signature in producer
  for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_node_producer->sbp_sig_list_.size();
       sbp_id_producer++) {
    sbp_edge->cost_[sbp_id_producer].resize(consumer_sbp_size, 0);
  }

  // Assemble copy cost from producer to proxy of producer
  OpNode* producer = sbp_node_producer->op_node_;

  // get parallel description. Number of devices.
  const ParallelDesc& producer_parallel_desc = producer->parallel_desc();
  // Need to be careful, the logical blob description should be independent to current
  // NdSbp. Use producer or op_node?
  const BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
  const std::string& obn = *CHECK_JUST(producer->op().obn4lbi(lbi));

  // A buffer to store the sbp parallel id
  std::vector<int32_t> sbp_parallel_ids;

  // look through sbp signature in producer
  for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_node_producer->sbp_sig_list_.size();
       sbp_id_producer++) {
    // get sbp parallel for a logical blob in producer
    const auto& producer_sbp_bn_in_op2sbp_parallel =
        sbp_node_producer->sbp_sig_list_[sbp_id_producer].bn_in_op2nd_sbp();
    const NdSbp& sbp_producer = producer_sbp_bn_in_op2sbp_parallel.at(obn);

    // look through sbp parallel set in consumer
    for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
      const BinarySet& sbp_parallel_set = sbp_proxy->parallel_candidates_[sbp_id_consumer];
      sbp_parallel_set.QuickOutput(sbp_parallel_ids);

      // look through all sbp parallels in a sbp parallel set
      for (int32_t sbp_parallel_id : sbp_parallel_ids) {
        // get sbp parallel for a logical blob in consumer
        const NdSbp& sbp_consumer = id2nd_sbp_[sbp_parallel_id];

        // compute copy cost for a specific logical blob
        // Use the parallel description of producer as those for consumer for now.
        sbp_edge->cost_[sbp_id_producer][sbp_id_consumer] +=
            CHECK_JUST(ComputeCopyCostWithMiddleNodes(sbp_producer, sbp_consumer, logical_blob_desc,
                                                      producer_parallel_desc,
                                                      producer_parallel_desc, /*is_same=*/false));
      }
    }
  }
}

// Initialize copy cost from proxy of producer to consumers
void SbpCollector::InitializeCopyCostFromProxy2Consumer(
    SbpNode* sbp_proxy,
    const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
    const HashMap<std::string, SbpNode*>& op_name2sbp_node) const {
  // Connect sbp proxy and consumers
  for (const auto& consumer_bn_group : consumer_bn2sbp_set) {
    // consumer in cost model
    SbpNode* sbp_node_consumer = op_name2sbp_node.find(consumer_bn_group.first.first)->second;
    // input blob name of logical blob in consumer
    const std::string& ibn = consumer_bn_group.first.second;

    // check is_mutable in consumer
    OpNode* consumer = sbp_node_consumer->op_node_;
    CHECK(!RequireSameSbp(consumer, ibn)) << "Create a proxy for an unsuitable consumer!\n";

    // Connect sbp proxy and consumer
    sbp_proxy->PointTo(sbp_node_consumer);
    // the sbp edge connecting proxy and consumer
    SbpEdge* sbp_edge = sbp_node_consumer->FindEdgeWithNode(sbp_proxy);
    sbp_edge->cost_.resize(sbp_proxy->parallel_candidates_.size());
    int32_t consumer_sbp_size = sbp_node_consumer->sbp_sig_list_.size();

    // look through sbp parallel set in proxy
    for (int32_t sbp_id_producer = 0; sbp_id_producer < sbp_proxy->parallel_candidates_.size();
         sbp_id_producer++) {
      // initialization for copy cost
      sbp_edge->cost_[sbp_id_producer].resize(consumer_sbp_size, 0);
      // get sbp parallel set for a logical blob in proxy
      BinarySet& parallel_candidate = sbp_proxy->parallel_candidates_[sbp_id_producer];

      // look through sbp signatures in consumers
      for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
        // get sbp parallel for a logical blob in consumer
        const auto& consumer_sbp_bn_in_op2sbp_parallel =
            sbp_node_consumer->sbp_sig_list_[sbp_id_consumer].bn_in_op2nd_sbp();
        const NdSbp& sbp_consumer = consumer_sbp_bn_in_op2sbp_parallel.at(ibn);

        if ((!parallel_candidate.CheckExistence(nd_sbp_universe_.find(sbp_consumer)->second))) {
          sbp_edge->cost_[sbp_id_producer][sbp_id_consumer] = GetMaxVal<float>();
        }
      }
    }
  }
}

// Export list of possible combination of Sbp Parallels
void SbpCollector::ProxySbpCandidate(const OpGraph& op_graph,
                                     const HashMap<std::string, SbpNode*>& op_name2sbp_node,
                                     SbpGraph& sbp_graph) {
  // If needed, we can output the mapping from operator name to its proxy.
  // HashMap<std::string, HashMap<LogicalBlobId, SbpNode*>>&
  //     op_name2lbi2sbp_proxy;

  // mapping from a logical blob id to index
  HashMap<LogicalBlobId, int32_t> lbi2index;
  // mapping from the index to producer, consumer and corresponding input blob name, possible sbp
  // sets
  std::vector<const OpNode*> index2producer;
  std::vector<std::unordered_set<int32_t>> index2sbp_set;
  // mapping from consumers and input blob names to an unordered_set of SBP Parallel.
  std::vector<HashMap<std::pair<std::string, std::string>, BinarySet>> index2consumer_bn2sbp_set;

  for (auto* consumer_sbp_node : sbp_graph.node_list_) {
    auto* node = consumer_sbp_node->op_node_;

    OperatorConf::OpTypeCase op_type_case = node->op().op_conf().op_type_case();
    // If not support boxing, just skip it.
    if (IsClassRegistered<int32_t, DisableInputBoxingGroup>(op_type_case)) { return; }
    for (const std::string& ibn : node->op().input_bns()) {
      // Skip those blobs who enforce same SBP.
      if (RequireSameSbp(node, ibn)) {
        // Enforcing same SBP. Can not collect sbp from this blob.
        continue;
      }

      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode& producer = node->ProducerOpNode4Lbi(lbi);

      // not building proxy for fixed operators
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
        index2consumer_bn2sbp_set.resize(index + 1);
        index2sbp_set.resize(index + 1);
      } else {
        index = iterator_lbi->second;
      }

      // a set to store the id of all possible SBP Parallel for a downstream op
      // should filter out repeated SBP Parallel by pre-storing them into an unordered_set
      BinarySet& nd_sbp_ids = index2consumer_bn2sbp_set[index][{node->op().op_name(), ibn}];
      nd_sbp_ids.Initialize(nd_sbp_universe_.size());
      // The union sbp set of all the consumers
      std::unordered_set<int32_t>& union_nd_sbp_ids = index2sbp_set[index];
      for (auto& sbp_sig : consumer_sbp_node->sbp_sig_list_) {
        const auto& map = sbp_sig.bn_in_op2nd_sbp();
        const auto& iter = map.find(ibn);
        CHECK(iter != map.end()) << "blob_name " << ibn << " not found in sbp signature";
        const NdSbp& consumer_sbp = iter->second;
        // filter out repeated SBP
        int32_t sbp_universe_id = nd_sbp_universe_.find(consumer_sbp)->second;
        nd_sbp_ids.AddEntry(sbp_universe_id);
        union_nd_sbp_ids.insert(sbp_universe_id);
      }
    }
  };

  // A set of binary set with broadcast only
  // std::unordered_set<BinarySet, BinarySetHasher> parallel_candidates_initializer;
  // BinarySet one_broadcast(nd_sbp_universe_.size());
  // one_broadcast.AddEntry(0);
  // parallel_candidates_initializer.insert(std::move(one_broadcast));

  // Decide if we should insert a proxy for each logical blob
  for (auto& lbi_index : lbi2index) {
    int32_t index = lbi_index.second;
    // Only insert proxy for those blobs with multiple downstream consumers.
    if (index2consumer_bn2sbp_set[index].size() < 2) { continue; }
    // Maximum number of possible sbp in the proxy
    int32_t max_num_sbp_proxy =
        std::min(max_num_sbp_proxy_, index2consumer_bn2sbp_set[index].size());
    // producer in cost model
    const std::string& producer_name = index2producer[index]->op().op_name();
    SbpNode* sbp_node_producer = op_name2sbp_node.find(producer_name)->second;

    const LogicalBlobId& lbi = lbi_index.first;
    // store all the binary sets of SBP Parallel into an unordered_set.
    // std::vector<BinarySet> parallel_candidates;

    // generate sbp proxy
    SbpNode* sbp_proxy = sbp_graph.GenerateNode();

    // A: {B, S0, S1, S2, S3}, C: {B, S0}, D: {B, S0}
    // {S1, S2, S3} show up only once, a parallel candidate should not contain two of them
    std::vector<BinarySet> unique_sbp_groups;
    FindUniqueSbpGroups(index2consumer_bn2sbp_set[index], index2sbp_set[index], accumulator_,
                        bs_buffer_, unique_sbp_groups);

    // Depth first search to collect Sbp Parallel information for the whole sbp set
    DfsSbpSet(0, max_num_sbp_proxy, index2sbp_set[index], index2sbp_set[index].begin(),
              index2consumer_bn2sbp_set[index], unique_sbp_groups, sbp_proxy->parallel_candidates_);

    // Initialize computation cost
    sbp_proxy->cost_.resize(sbp_proxy->parallel_candidates_.size(), 0);

    // Transfer a logical blob from producer to a sbp proxy of this blob
    sbp_node_producer->PointTo(sbp_proxy);

    // Compute copy cost between producer and proxy
    InitializeCopyCostFromNode2Proxy(sbp_proxy, lbi);

    // Build connection and compute copy cost between proxy and consumers
    InitializeCopyCostFromProxy2Consumer(sbp_proxy, index2consumer_bn2sbp_set[index],
                                         op_name2sbp_node);

    // Unloading
    for (const auto& consumer_bn_group : index2consumer_bn2sbp_set[index]) {
      // consumer in cost model
      SbpNode* sbp_node_consumer = op_name2sbp_node.find(consumer_bn_group.first.first)->second;
      // the sbp edge connecting producer and consumer
      SbpEdge* edge_found = sbp_node_consumer->FindEdgeWithNode(sbp_node_producer);
      // unload logical blob from sbp edges
      edge_found->UnloadLbi(lbi);
      // Do not clip this edge. Save it for wait time.
      // clip this edge if it no longer carries any blob
      // We don't clip edges before since we have transfer cost
      // Now we clip edges, which makes the topology simpler
      if (edge_found->EmptyLbi() && edge_found->wait_time_ <= 0.0
          && edge_found->wait_time_ > -0.5) {
        sbp_graph.ClipEdge(edge_found);
      }
    }
  }
}

// Depth first search to collect Sbp Parallel information for different logical blob ids
void SbpCollector::DfsSbpSet(
    int32_t depth, int32_t max_depth, const std::unordered_set<int32_t>& sbp_sets,
    const std::unordered_set<int32_t>::iterator& start_it,
    const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
    const std::vector<BinarySet>& unique_sbp_groups, std::vector<BinarySet>& parallel_candidates) {
  if (depth > 0) {
    if (IfIntersectAll(consumer_bn2sbp_set, bs_buffer_)
        && No2SbpFromSameUniqueGroup(bs_buffer_, unique_sbp_groups)) {
      // store the binary set into an unordered_set
      parallel_candidates.push_back(bs_buffer_);
    }
  }
  if (depth >= max_depth) { return; }

  // go through the rest of the sbp parallel
  std::unordered_set<int32_t>::iterator curr_it = start_it;
  while (curr_it != sbp_sets.end()) {
    // Take the value out
    int32_t nd_sbp_num = *curr_it;
    // Then move to the next pointer
    ++curr_it;
    if (accumulator_[nd_sbp_num] == 0) {
      bs_buffer_.AddEntry(nd_sbp_num);
      ++accumulator_[nd_sbp_num];
      DfsSbpSet(depth + 1, max_depth, sbp_sets, curr_it, consumer_bn2sbp_set, unique_sbp_groups,
                parallel_candidates);
      bs_buffer_.DeleteEntry(nd_sbp_num);
      --accumulator_[nd_sbp_num];
    }
  }
}

}  // namespace auto_parallel

}  // namespace oneflow
