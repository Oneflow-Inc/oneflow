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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_EDGE_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_EDGE_H_

#include <assert.h>
#include <algorithm>
#include <unordered_set>

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {
namespace auto_parallel {

template<class SbpSignature>
class SbpGraph;

template<class SbpSignature>
class SbpEdge final {
  /* There are 3 types of edges:
   * 1. start_node_ -> end_node_
   *      Nothing special
   * 2. Multiple start_node_ -> end_node_
   *      edge_list_ will store all the edges which goes from start_node_ to end_node_
   * 3. start_node_ -> mid_node_ -> end_node_
   *      It will pass by a middle node.
   */
 public:
  // Constructor for type 1 & 2
  SbpEdge(SbpNode<SbpSignature>* start_node, SbpNode<SbpSignature>* end_node)
      : start_node_(start_node), end_node_(end_node) {
    mid_node_ = nullptr;
  }
  // Constructor for type 3
  SbpEdge(SbpNode<SbpSignature>* start_node, SbpNode<SbpSignature>* mid_node,
          SbpNode<SbpSignature>* end_node, SbpEdge<SbpSignature>* first_edge,
          SbpEdge<SbpSignature>* second_edge);

  // Deconstructor
  ~SbpEdge() {
    if (mid_node_ != nullptr) { delete mid_node_; }
    for (auto& this_edge : edge_list_) { delete this_edge; }
  }

  // Compute copy cost for type 1
  void ComputeCost(
      const std::function<double(SbpNode<SbpSignature>*, SbpSignature*, SbpNode<SbpSignature>*,
                                 SbpSignature*)>& SbpInferHint4Ibn);

  // Update copy cost for type 2 and 3
  void SummarizeCost();
  // Duplicate Cost. Designed for merging two nodes.
  void DuplicateCost(bool ifStart, bool ifFirst,
                     const std::vector<std::pair<int32_t, int32_t>>& mergedSigId2ChildrenSigId);
  // Determine Final SbpSignature for attachment of this edge
  void FinalizeSbp();
  // Use Greedy Strategy to pick the sbp signature with minimum cost for this
  // edge. You should have an initial strategy before running this. And the
  // graph should be fully eliminated.
  double GreedyStrategy();

  // a set of ids of logical blobs carried/transferred on this sbp edge
  std::unordered_set<oneflow::LogicalBlobId> CarryLbis;

  // load a logical blob
  void LoadLbi(oneflow::LogicalBlobId lbi) { CarryLbis.insert(lbi); }

  // check existancy of a logical blob
  bool SearchLbi(oneflow::LogicalBlobId lbi) const {
    return CarryLbis.find(lbi) != CarryLbis.end();
  }

  // unload a logical blob
  void UnloadLbi(oneflow::LogicalBlobId lbi) {
    if (CarryLbis.erase(lbi) == 0) { std::cout << "Unload an empty lbi!" << std::endl; }
  }

  // Not carrying any blob
  bool EmptyLbi() const { return CarryLbis.empty(); }

  // Get the minimum element in Cost
  double GetMinCost();
  // Get the maximum element in Cost
  double GetMaxCost();

  // Detect and spread overlaps for this edge and start_node_ if it is a proxy of lbi.
  void DetectSpreadOverlap(double overlap_ratio_);

  // Adjust cost with overlaps
  void AdjustOverlapCost();

  // Assemble copy cost
  // compute_cost = true: It is computing cost
  // compute_cost = false: It is deciding whether this edge needs the wait time.
  void InitializeCopyCost(const std::string& ibn, bool compute_cost, bool use_sbp_collector_);

  // find the cut ratio
  // (#c>cut_cost in Cost)/(#c in Cost)
  // But we would lift the cut ratio to 1 to filter out some improper couples
  double FindCutRatio(int32_t threshold);
  // Get the cut ratio
  double GetCutRatio();

 private:
  friend class SbpNode<SbpSignature>;
  friend class SbpGraph<SbpSignature>;
  friend class SbpCollector;
  friend class SbpConstructor;

  // The edge point from start_node_ to end_node_
  // It will have a middle node if and only if type 3
  SbpNode<SbpSignature>*start_node_, *mid_node_, *end_node_;
  // Cost[sbp_i][sbp_j] is the total cost from start_node_ with sbp_i to end_node_
  // with sbp_j
  std::vector<std::vector<double>> cost_;
  // SbpSignature for mid_node_ with corresponding Cost if type 3, empty otherwise
  std::vector<std::vector<int32_t>> mid_node_sbp_sig_;
  // Contained edge list:
  // empty if type 1,
  // Parallel edges if type 2,
  // succeed edges if type 3
  // the edge list might have reverse direction:
  // example 1: type 3 edge_list_ contain two edges:
  //        mid_node_ -> start_node_, mid_node_ -> end_node_;
  // example 2: type 2 edge_list_ contain three edges:
  //        start_node_ -> end_node_, end_node_ -> start_node_, start_node_ -> end_node_;
  std::vector<SbpEdge<SbpSignature>*> edge_list_;
  // Time waiting for other gpus. pthread_cond_wait
  double wait_time_ = -1.0;

  // Mininum and maximum cost would not be changed by eliminations, which will generate new edges.
  // Also would not be changed by node merging, which will only perform cost copy for the expanding
  // dimensions.
  // Minimum cost in the 2D array Cost.
  // Would be initialized after GetMinCost();
  // Only used in the final graph.
  double min_cost_ = -1.0;
  // Maximum cost in the 2D array Cost.
  // Would be initialized after GetMaxCost();
  // Only used in the original graph.
  // double max_cost = -1.0;
  // overlap ratio. Applied in copy cost.
  double overlap_ratio_ = 1.0;
};

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.
template<class SbpSignature>
SbpEdge<SbpSignature>::SbpEdge(SbpNode<SbpSignature>* start_node, SbpNode<SbpSignature>* mid_node,
                               SbpNode<SbpSignature>* end_node, SbpEdge<SbpSignature>* first_edge,
                               SbpEdge<SbpSignature>* second_edge)
    : start_node_(start_node), mid_node_(mid_node), end_node_(end_node) {
  edge_list_.emplace_back(first_edge);
  edge_list_.emplace_back(second_edge);
};

template<class SbpSignature>
void SbpEdge<SbpSignature>::ComputeCost(
    const std::function<double(SbpNode<SbpSignature>*, SbpSignature*, SbpNode<SbpSignature>*,
                               SbpSignature*)>& SbpInferHint4Ibn) {
  cost_.resize(start_node_->Cost.size());
  int32_t EndNodeSbpSize = end_node_->cost_.size();
  for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
    cost_[sbp_start].resize(EndNodeSbpSize);
    for (int32_t sbp_end = 0; sbp_end < EndNodeSbpSize; sbp_end++) {
      cost_[sbp_start][sbp_end] =
          SbpInferHint4Ibn(start_node_, start_node_->SbpSignatureList[sbp_start], end_node_,
                           end_node_->SbpSignatureList[sbp_end]);
    }
  }
};

template<class SbpSignature>
void SbpEdge<SbpSignature>::SummarizeCost() {
  if (mid_node_) {
    cost_.resize(start_node_->Cost.size());
    mid_node_sbp_sig_.resize(start_node_->Cost.size());
    int32_t EndNodeSbpSize = end_node_->Cost.size();
    int32_t MidNodeSbpSize = mid_node_->Cost.size();
    for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
      cost_[sbp_start].resize(EndNodeSbpSize);
      mid_node_sbp_sig_[sbp_start].resize(EndNodeSbpSize);
      for (int32_t sbp_end = 0; sbp_end < EndNodeSbpSize; sbp_end++) {
        for (int32_t sbp_mid = 0; sbp_mid < MidNodeSbpSize; sbp_mid++) {
          // Add middle node cost
          double temp = mid_node_->Cost[sbp_mid];
          // Add first edge cost
          if (edge_list_[0]->start_node_ == start_node_) {
            temp += edge_list_[0]->cost_[sbp_start][sbp_mid];
          } else {
            temp += edge_list_[0]->cost_[sbp_mid][sbp_start];
          }
          // Add second edge cost
          if (edge_list_[1]->end_node_ == end_node_) {
            temp += edge_list_[1]->cost_[sbp_mid][sbp_end];
          } else {
            temp += edge_list_[1]->cost_[sbp_end][sbp_mid];
          }

          // Compare and look for the minimum cost
          if (sbp_mid == 0) {
            cost_[sbp_start][sbp_end] = temp;
            mid_node_sbp_sig_[sbp_start][sbp_end] = sbp_mid;
          } else if (temp < cost_[sbp_start][sbp_end]) {
            cost_[sbp_start][sbp_end] = temp;
            mid_node_sbp_sig_[sbp_start][sbp_end] = sbp_mid;
          }
        }
      }
    }
  } else {
    cost_.resize(start_node_->Cost.size());
    int32_t EndNodeSbpSize = end_node_->Cost.size();
    for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
      cost_[sbp_start].resize(EndNodeSbpSize);
      for (int32_t sbp_end = 0; sbp_end < EndNodeSbpSize; sbp_end++) {
        cost_[sbp_start][sbp_end] = 0;
        for (int32_t edge_num = 0; edge_num < edge_list_.size(); edge_num++) {
          if (edge_list_[edge_num]->start_node_ == start_node_) {
            cost_[sbp_start][sbp_end] += edge_list_[edge_num]->cost_[sbp_start][sbp_end];
          } else {
            cost_[sbp_start][sbp_end] += edge_list_[edge_num]->cost_[sbp_end][sbp_start];
          }
        }
      }
    }
  }
}

template<class SbpSignature>
void SbpEdge<SbpSignature>::DuplicateCost(
    bool ifStart, bool ifFirst,
    const std::vector<std::pair<int32_t, int32_t>>& mergedSigId2ChildrenSigId) {
  const int32_t num_sig = mergedSigId2ChildrenSigId.size();
  std::vector<std::vector<double>> temp_cost;
  std::vector<std::vector<int32_t>> tmpMidNodeSbpSig;
  if (ifStart) {
    temp_cost.resize(num_sig);
    if (mid_node_) { tmpMidNodeSbpSig.resize(num_sig); }
    for (int32_t i = 0; i < num_sig; i++) {
      const int32_t sig_idx =
          ifFirst ? mergedSigId2ChildrenSigId[i].first : mergedSigId2ChildrenSigId[i].second;
      temp_cost[i] = cost_[sig_idx];
      if (mid_node_) { tmpMidNodeSbpSig[i] = mid_node_sbp_sig_[sig_idx]; }
    }
  } else {
    const int32_t num_start_sig = cost_.size();
    temp_cost.resize(num_start_sig);
    if (mid_node_) { tmpMidNodeSbpSig.resize(num_start_sig); }
    for (int32_t i = 0; i < num_start_sig; i++) {
      temp_cost[i].resize(num_sig);
      if (mid_node_) { tmpMidNodeSbpSig[i].resize(num_sig); }
      for (int32_t j = 0; j < num_sig; j++) {
        const int32_t sig_idx =
            ifFirst ? mergedSigId2ChildrenSigId[j].first : mergedSigId2ChildrenSigId[j].second;
        temp_cost[i][j] = cost_[i][sig_idx];
        if (mid_node_) { tmpMidNodeSbpSig[i][j] = mid_node_sbp_sig_[i][sig_idx]; }
      }
    }
  }

  cost_ = temp_cost;
  if (mid_node_) { mid_node_sbp_sig_ = tmpMidNodeSbpSig; }
}

template<class SbpSignature>
void SbpEdge<SbpSignature>::FinalizeSbp() {
#ifdef DEBUG_ALGORITHM_
  // printf("Finalize Sbp edge: from %d to %d, EdgeListNum: %d\n",
  // start_node_->id,
  //       end_node_->id, edge_list_.size());
#endif  // DEBUG_ALGORITHM_

  // Finalize Sbp for mid_node_
  if (mid_node_) {
    mid_node_->FinalSbpSignatureId =
        mid_node_sbp_sig_[start_node_->FinalSbpSignatureId][end_node_->FinalSbpSignatureId];
    mid_node_->FinalizeSbp();
  }
  for (const auto& this_edge : edge_list_) { this_edge->FinalizeSbp(); }
}

template<class SbpSignature>
double SbpEdge<SbpSignature>::GreedyStrategy() {
  // Sbp combination of the minimum cost
  int32_t MinSbpStart = start_node_->FinalSbpSignatureId,
          MinSbpEnd = end_node_->FinalSbpSignatureId;
  // An unordered_map to evaluate cost between two edge nodes and other nodes.
  std::unordered_map<int32_t, int32_t> EdgeNodeListIds = {{start_node_->NodeListId, 0},
                                                          {end_node_->NodeListId, 1}};
  // pre-compute and store the Current Cost for end_node_.
  std::vector<double> EndNodeCurrCost(end_node_->Cost.size());
  for (int32_t sbp_end = 0; sbp_end < cost_[0].size(); sbp_end++) {
    end_node_->FinalSbpSignatureId = sbp_end;
    EndNodeCurrCost[sbp_end] = end_node_->EvalOutNbhCost(EdgeNodeListIds);
  }
  // pre-copmpute and store the Current Cost for start_node_.
  std::vector<double> StartNodeCurrCost(start_node_->Cost.size());
  for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
    start_node_->FinalSbpSignatureId = sbp_start;
    StartNodeCurrCost[sbp_start] = start_node_->EvalOutNbhCost(EdgeNodeListIds);
  }
  // Current Cost, Minimum Cost, Cost with original sbp
  double CurrCost, MinCost, OrgCost;
  MinCost =
      StartNodeCurrCost[MinSbpStart] + EndNodeCurrCost[MinSbpEnd] + cost_[MinSbpStart][MinSbpEnd];
  OrgCost = MinCost;

  for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
    for (int32_t sbp_end = 0; sbp_end < cost_[0].size(); sbp_end++) {
      // compute Current Cost for Neighborhood of edge
      end_node_->FinalSbpSignatureId = sbp_end;
      CurrCost =
          StartNodeCurrCost[sbp_start] + EndNodeCurrCost[sbp_end] + cost_[sbp_start][sbp_end];
      // Find the minimum current cost
      if (CurrCost < MinCost) {
        MinCost = CurrCost;
        MinSbpStart = sbp_start;
        MinSbpEnd = sbp_end;
      }
    }
  }
  start_node_->FinalSbpSignatureId = MinSbpStart;
  end_node_->FinalSbpSignatureId = MinSbpEnd;
  return MinCost - OrgCost;
}

// Get the minimum element in Cost
template<class SbpSignature>
double SbpEdge<SbpSignature>::GetMinCost() {
  // used the stored value if pre-computed.
  if (min_cost_ >= 0) { return min_cost_; }
  // Check the size of Cost
  CHECK(cost_.size() > 0) << "Cost not initialized!" << std::endl;
  // Compute the min_cost
  min_cost_ = *std::min_element(cost_[0].begin(), cost_[0].end());
  for (int32_t i = 1; i < cost_.size(); i++) {
    double min_cost_row = *std::min_element(cost_[i].begin(), cost_[i].end());
    if (min_cost_row < min_cost_) { min_cost_ = min_cost_row; }
  }
  return min_cost_;
}

// Get the maximum element in Cost
template<class SbpSignature>
double SbpEdge<SbpSignature>::GetMaxCost() {
  // used the stored value if pre-computed.
  // if (max_cost >= 0) return max_cost;
  // Check the size of Cost
  CHECK(cost_.size() > 0) << "Cost not initialized!" << std::endl;
  // Compute the max_cost
  double max_cost = -1.0;
  for (int32_t i = 0; i < cost_.size(); i++) {
    for (int32_t j = 0; j < cost_[i].size(); j++) {
      if (cost_[i][j] < cut_cost && cost_[i][j] > max_cost) { max_cost = cost_[i][j]; }
    }
  }
  return max_cost;
}

// Detect and spread overlaps for this edge and start_node_ if it is a proxy of lbi.
template<class SbpSignature>
void SbpEdge<SbpSignature>::DetectSpreadOverlap(double overlap_ratio) {
  if (overlap_ratio < 1.0) {
    if (overlap_ratio < 0.0) { overlap_ratio = 0.0; }

    if (start_node_->op_node) {
      // change overlap ratio for a normal edge
      // We could use the minimum or multiplication here.
      // To be noted that sbp_proxy may have multiple outcoming edges, we can not adjust overlap
      // cost right away.
      if (overlap_ratio < overlap_ratio_) { overlap_ratio_ = overlap_ratio; }
    } else {
      // For an edge with a proxy start node, do not change the overlap ratio since the cost only
      // contains 0 and 3e38. Change the overlap ratio for the previous edge.
      start_node_->DetectSpreadOverlap(overlap_ratio_);
    }
  }
}

// Adjust cost with overlaps
template<class SbpSignature>
void SbpEdge<SbpSignature>::AdjustOverlapCost() {
  if (overlap_ratio_ >= 1.0) { return; }
  if (overlap_ratio_ < 0.0) { overlap_ratio_ = 0.0; }
  for (int32_t i = 0; i < cost_.size(); i++) {
    for (int32_t j = 0; j < cost_[i].size(); j++) {
      if (cost_[i][j] > 0.0 && cost_[i][j] < cut_cost) {
        cost_[i][j] = overlap_ratio_ * cost_[i][j];
      }
    }
  }
}

// Assemble copy cost
template<class SbpSignature>
void SbpEdge<SbpSignature>::InitializeCopyCost(const std::string& ibn, bool compute_cost,
                                               bool use_sbp_collector_) {
  // In this part, we assemble the cost from nodes to nodes.
  if (start_node_->op_node && end_node_->op_node) {
    oneflow::OpNode* consumer = end_node_->op_node;

    // Add copy cost for each blob
    const oneflow::LogicalBlobId& lbi = consumer->op().BnInOp2Lbi(ibn);

    // Check whether lbi is transferred by this edge
    if (use_sbp_collector_ && compute_cost && !SearchLbi(lbi)) { return; }

    oneflow::OpNode* producer = start_node_->op_node;
    const std::string& producer_lbn = *CHECK_JUST(producer->op().obn4lbi(lbi));
    const oneflow::ParallelDesc& producer_parallel_desc =
        *CHECK_JUST(producer->op().GetParallelDesc4BnInOp(producer_lbn));
    const oneflow::ParallelDesc& consumer_parallel_desc =
        *CHECK_JUST(consumer->op().GetParallelDesc4BnInOp(ibn));

    // Need to be careful, the logical blob description should be independent to current
    // SbpParallel. Use producer or op_node?
    const oneflow::BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
    const std::string& obn = *CHECK_JUST(producer->op().obn4lbi(lbi));
    // If we are deciding whether we need the wait time, then make is_same_sbp true.
    // B->S cause cudaEventSynchronize in current implementation.
    bool is_same_sbp = (!compute_cost) || IsSameSbp(consumer, ibn);
    int32_t consumer_sbp_size = end_node_->SbpSignatureList.size();
    LazyMode::Guard enable_lazy_mode(true);

    // look through sbp signature in producer
    for (int32_t sbp_id_producer = 0; sbp_id_producer < start_node_->SbpSignatureList.size();
         sbp_id_producer++) {
      // get sbp parallel for a logical blob in producer
      const auto producer_sbp_bn_in_op2sbp_parallel =
          start_node_->SbpSignatureList[sbp_id_producer]->bn_in_op2nd_sbp();
      const NdSbp& sbp_producer = producer_sbp_bn_in_op2sbp_parallel.at(obn);

      // look through sbp signature in consumer
      for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
        // get sbp parallel for a logical blob in consumer
        const auto consumer_sbp_bn_in_op2sbp_parallel =
            end_node_->SbpSignatureList[sbp_id_consumer]->bn_in_op2nd_sbp();
        const NdSbp& sbp_consumer = consumer_sbp_bn_in_op2sbp_parallel.at(ibn);

        // compute copy cost for a specific logical blob
        cost_[sbp_id_producer][sbp_id_consumer] += CHECK_JUST(ComputeCopyCostWithMiddleNodes(
            sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
            consumer_parallel_desc, is_same_sbp));
      }
    }
  }
}

// Set the cut ratio
template<class SbpSignature>
double SbpEdge<SbpSignature>::GetCutRatio() {
  int32_t num = 0;
  for (int32_t i = 0; i < cost_.size(); i++) {
    for (int32_t j = 0; j < cost_[i].size(); j++) {
      if (cost_[i][j] < cut_cost) { num++; }
    }
  }
  return double(num) / double(cost_.size() * cost_[0].size());
}

// find the cut ratio
// (#c>cut_cost in Cost)/(#c in Cost)
template<class SbpSignature>
double SbpEdge<SbpSignature>::FindCutRatio(int32_t threshold) {
  double cut_ratio = GetCutRatio();
  // lift the cut ratio to 1 to filter out some improper couples to avoid unlimited merging
  double n = cost_.size();
  double m = cost_[0].size();
  double num = cut_ratio * n * m;
  cut_ratio += 0.16 * (n + m) / double(threshold);
  if (num <= n * 2 || num <= m * 2 || (num <= threshold && cut_ratio < 0.51)) {
    return cut_ratio;
  } else {
    return 1.0;
  }
}

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_EDGE_H_
