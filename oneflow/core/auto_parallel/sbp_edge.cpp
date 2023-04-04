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

#include <assert.h>
#include <algorithm>
#include <unordered_set>
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/auto_parallel/sbp_edge.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_graph.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {
namespace auto_parallel {

extern double kMemoryRatio;

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.
SbpEdge::SbpEdge(SbpNode* start_node, SbpNode* mid_node, SbpNode* end_node, SbpEdge* first_edge,
                 SbpEdge* second_edge)
    : start_node_(start_node), mid_node_(mid_node), end_node_(end_node) {
  // The first edge must between start_node and mid_node, but it could be
  // start_node -> mid_node or mid_node -> start node
  // Same for the second edge.
  edge_list_.emplace_back(first_edge);
  edge_list_.emplace_back(second_edge);
};

// Deconstructor
SbpEdge::~SbpEdge() {
  if (mid_node_ != nullptr) { delete mid_node_; }
  for (auto& this_edge : edge_list_) { delete this_edge; }
}

void SbpEdge::SummarizeCost() {
  // If any sub data structure is in the memory support,
  // then this edge is in the memory support
  if (mid_node_ && mid_node_->in_memory_support_) {
    in_memory_support_ = true;
  } else {
    in_memory_support_ = std::any_of(edge_list_.begin(), edge_list_.end(), [](SbpEdge* sbp_edge) {
      return sbp_edge->in_memory_support_;
    });
  }
  // We would need to compute the memory for this elimination
  int32_t start_node_sbp_size = start_node_->weighted_cost_.size();
  if (in_memory_support_) { memory_.resize(start_node_sbp_size); }
  weighted_cost_.resize(start_node_sbp_size);
  // Copy cost and memory cost
  if (mid_node_) {
    // Buffer
    int64_t memory_cost = 0;
    int64_t min_memory_cost = 0;
    int32_t min_sbp_mid = 0;
    double weighted_cost = 0.0;
    double min_weighted_cost = 0.0;
    // Node elimination
    mid_node_sbp_sig_.resize(start_node_sbp_size);
    int32_t end_node_sbp_size = end_node_->weighted_cost_.size();
    int32_t mid_node_sbp_size = mid_node_->weighted_cost_.size();
    for (int32_t sbp_start = 0; sbp_start < start_node_sbp_size; sbp_start++) {
      if (in_memory_support_) { memory_[sbp_start].resize(end_node_sbp_size); }
      weighted_cost_[sbp_start].resize(end_node_sbp_size);
      mid_node_sbp_sig_[sbp_start].resize(end_node_sbp_size);
      for (int32_t sbp_end = 0; sbp_end < end_node_sbp_size; sbp_end++) {
        for (int32_t sbp_mid = 0; sbp_mid < mid_node_sbp_size; sbp_mid++) {
          // Add middle node cost
          memory_cost = mid_node_->GetMemory(sbp_mid);
          weighted_cost = mid_node_->weighted_cost_[sbp_mid];
          // Add first edge cost
          if (edge_list_[0]->end_node_ == mid_node_) {
            int32_t edge_sbp_start =
                start_node_->GetComponentSbpId(sbp_start, edge_list_[0]->start_node_);
            memory_cost += edge_list_[0]->GetMemory(edge_sbp_start, sbp_mid);
            weighted_cost += edge_list_[0]->weighted_cost_[edge_sbp_start][sbp_mid];
          } else {
            int32_t edge_sbp_start =
                start_node_->GetComponentSbpId(sbp_start, edge_list_[0]->end_node_);
            memory_cost += edge_list_[0]->GetMemory(sbp_mid, edge_sbp_start);
            weighted_cost += edge_list_[0]->weighted_cost_[sbp_mid][edge_sbp_start];
          }
          // Add second edge cost
          if (edge_list_[1]->start_node_ == mid_node_) {
            int32_t edge_sbp_end = end_node_->GetComponentSbpId(sbp_end, edge_list_[1]->end_node_);
            memory_cost += edge_list_[1]->GetMemory(sbp_mid, edge_sbp_end);
            weighted_cost += edge_list_[1]->weighted_cost_[sbp_mid][edge_sbp_end];
          } else {
            int32_t edge_sbp_end =
                end_node_->GetComponentSbpId(sbp_end, edge_list_[1]->start_node_);
            memory_cost += edge_list_[1]->GetMemory(edge_sbp_end, sbp_mid);
            weighted_cost += edge_list_[1]->weighted_cost_[edge_sbp_end][sbp_mid];
          }

          // Compare and look for the minimum cost
          if (sbp_mid == 0 || weighted_cost < min_weighted_cost) {
            min_sbp_mid = sbp_mid;
            min_memory_cost = memory_cost;
            min_weighted_cost = weighted_cost;
          }
        }
        // Store the results of the dynamic programming for minimizing the weighted sum
        if (in_memory_support_) { memory_[sbp_start][sbp_end] = min_memory_cost; }
        weighted_cost_[sbp_start][sbp_end] = min_weighted_cost;
        mid_node_sbp_sig_[sbp_start][sbp_end] = min_sbp_mid;
      }
    }
  } else {
    // Edge elimination
    int32_t end_node_sbp_size = end_node_->weighted_cost_.size();
    for (int32_t sbp_start = 0; sbp_start < weighted_cost_.size(); sbp_start++) {
      if (in_memory_support_) { memory_[sbp_start].resize(end_node_sbp_size); }
      weighted_cost_[sbp_start].resize(end_node_sbp_size);
      for (int32_t sbp_end = 0; sbp_end < end_node_sbp_size; sbp_end++) {
        int64_t memory_cost = 0;
        double weighted_cost = 0.0;
        for (int32_t edge_num = 0; edge_num < edge_list_.size(); edge_num++) {
          // For normal edge elimination, instead of recomputation with different memory ratio
          // Either (start_node_ == edge_list_[edge_num]->start_node_
          // and end_node_ == edge_list_[edge_num]->end_node_) is true
          // Or (start_node_ == edge_list_[edge_num]->end_node_ and
          // end_node_ == edge_list_[edge_num]->start_node_) is true.
          // At this moment, start_node_->component2merged_sig_id2component_sig_id_ is not
          // initialized. As a result, if start_node_ != edge_list_[edge_num]->start_node_,
          // IsComponent() would return false immediately.
          if (start_node_->IsComponent(edge_list_[edge_num]->start_node_)) {
            int32_t edge_sbp_start =
                start_node_->GetComponentSbpId(sbp_start, edge_list_[edge_num]->start_node_);
            int32_t edge_sbp_end =
                end_node_->GetComponentSbpId(sbp_end, edge_list_[edge_num]->end_node_);
            memory_cost += edge_list_[edge_num]->GetMemory(edge_sbp_start, edge_sbp_end);
            weighted_cost += edge_list_[edge_num]->weighted_cost_[edge_sbp_start][edge_sbp_end];
          } else {
            // At this moment
            // start_node_->IsComponent(edge_list_[edge_num]->end_node_)
            // end_node_->IsComponent(edge_list_[edge_num]->start_node_)
            int32_t edge_sbp_start =
                start_node_->GetComponentSbpId(sbp_start, edge_list_[edge_num]->end_node_);
            int32_t edge_sbp_end =
                end_node_->GetComponentSbpId(sbp_end, edge_list_[edge_num]->start_node_);
            memory_cost += edge_list_[edge_num]->GetMemory(edge_sbp_end, edge_sbp_start);
            weighted_cost += edge_list_[edge_num]->weighted_cost_[edge_sbp_end][edge_sbp_start];
          }
        }
        if (in_memory_support_) { memory_[sbp_start][sbp_end] = memory_cost; }
        weighted_cost_[sbp_start][sbp_end] = weighted_cost;
      }
    }
  }
}

void SbpEdge::DuplicateCost(
    bool merged_node_is_start_node, bool duplicating_first_node,
    const std::vector<std::pair<int32_t, int32_t>>& merged_sig_id2half_sig_id) {
  const int32_t num_sig = merged_sig_id2half_sig_id.size();
  std::vector<std::vector<double>> copy_cost;
  std::vector<std::vector<int32_t>> temp_mid_node_sbp_sig;
  std::vector<std::vector<int64_t>> temp_memory;
  std::vector<std::vector<double>> weighted_cost;
  if (merged_node_is_start_node) {
    if (edge_list_.empty()) { copy_cost.resize(num_sig); }
    if (mid_node_) { temp_mid_node_sbp_sig.resize(num_sig); }
    weighted_cost.resize(num_sig);
    if (in_memory_support_) { temp_memory.resize(num_sig); }
    for (int32_t i = 0; i < num_sig; i++) {
      const int32_t sig_idx = duplicating_first_node ? merged_sig_id2half_sig_id[i].first
                                                     : merged_sig_id2half_sig_id[i].second;
      if (edge_list_.empty()) { copy_cost[i] = cost_[sig_idx]; }
      weighted_cost[i] = weighted_cost_[sig_idx];
      if (mid_node_) { temp_mid_node_sbp_sig[i] = mid_node_sbp_sig_[sig_idx]; }
      if (in_memory_support_) { temp_memory[i] = memory_[sig_idx]; }
    }
  } else {
    const int32_t num_start_sig = weighted_cost_.size();
    if (edge_list_.empty()) { copy_cost.resize(num_start_sig); }
    weighted_cost.resize(num_start_sig);
    if (mid_node_) { temp_mid_node_sbp_sig.resize(num_start_sig); }
    if (in_memory_support_) { temp_memory.resize(num_start_sig); }
    for (int32_t i = 0; i < num_start_sig; i++) {
      if (edge_list_.empty()) { copy_cost[i].resize(num_sig); }
      weighted_cost[i].resize(num_sig);
      if (mid_node_) { temp_mid_node_sbp_sig[i].resize(num_sig); }
      if (in_memory_support_) { temp_memory[i].resize(num_sig); }
      for (int32_t j = 0; j < num_sig; j++) {
        const int32_t sig_idx = duplicating_first_node ? merged_sig_id2half_sig_id[j].first
                                                       : merged_sig_id2half_sig_id[j].second;
        if (edge_list_.empty()) { copy_cost[i][j] = cost_[i][sig_idx]; }
        weighted_cost[i][j] = weighted_cost_[i][sig_idx];
        if (mid_node_) { temp_mid_node_sbp_sig[i][j] = mid_node_sbp_sig_[i][sig_idx]; }
        if (in_memory_support_) { temp_memory[i][j] = memory_[i][sig_idx]; }
      }
    }
  }

  if (edge_list_.empty()) { cost_ = copy_cost; }
  weighted_cost_ = weighted_cost;
  if (mid_node_) { mid_node_sbp_sig_ = temp_mid_node_sbp_sig; }
  if (in_memory_support_) { memory_ = temp_memory; }
}

// Compute the weighted sum of the time and memory cost
void SbpEdge::ComputeWeightedCost() {
  if (edge_list_.empty()) {
    // If this edge does not contain any sub edges, it should have original cost
    weighted_cost_ = cost_;
    if (in_memory_support_) {
      for (int32_t i = 0; i < memory_.size(); i++) {
        auto& memory_i = memory_[i];
        auto& weighted_cost_i = weighted_cost_[i];
        for (int32_t j = 0; j < memory_[i].size(); j++) {
          weighted_cost_i[j] += kMemoryRatio * memory_i[j];
        }
      }
    }
  } else {
    // Compute the weighted cost for sub components
    for (auto& sbp_edge : edge_list_) { sbp_edge->ComputeWeightedCost(); }
    if (mid_node_) { mid_node_->ComputeWeightedCost(); }
    // Generate relationship if two vertices are merged nodes
    // For example, we have 4 nodes: A, B, C, D
    // and two edges: 1: A->B, 2: A->B
    // We merge the two edges 1 and 2 into 3: A->B.
    // Then we merge A and C into E and merge B and D into F.
    // Now the edge 3: E->F has two sub edges: 1: A->B, 2:A->B,
    // which tell us that the sub edges might have different vertices from the current edge.
    start_node_->GenerateComponentRelationship();
    end_node_->GenerateComponentRelationship();
    // Re-compute the weighted cost
    SummarizeCost();
  }
}

void SbpEdge::FinalizeSbp() {
  // Finalize Sbp for mid_node_
  if (mid_node_) {
    mid_node_->final_sbp_sig_id_ =
        mid_node_sbp_sig_[start_node_->final_sbp_sig_id_][end_node_->final_sbp_sig_id_];
    mid_node_->FinalizeSbp();
  }
  for (const auto& this_edge : edge_list_) { this_edge->FinalizeSbp(); }
}

double SbpEdge::GreedyStrategy() {
  // Sbp combination of the minimum cost
  int32_t min_sbp_start = start_node_->final_sbp_sig_id_,
          min_sbp_end = end_node_->final_sbp_sig_id_;
  // An unordered_map to evaluate cost between two edge nodes and other nodes.
  std::unordered_map<int32_t, int32_t> node_list_id2nbh_id = {{start_node_->node_list_id_, 0},
                                                              {end_node_->node_list_id_, 1}};
  // pre-compute and store the current cost between end_node_ and outside.
  std::vector<double> end_node_out_cost(end_node_->weighted_cost_.size());
  for (int32_t sbp_end = 0; sbp_end < weighted_cost_[0].size(); sbp_end++) {
    end_node_->final_sbp_sig_id_ = sbp_end;
    end_node_out_cost[sbp_end] = end_node_->EvalOutNbhCost(node_list_id2nbh_id);
  }
  // pre-compute and store the current cost between start_node_ and outside.
  std::vector<double> start_node_out_cost(start_node_->weighted_cost_.size());
  for (int32_t sbp_start = 0; sbp_start < weighted_cost_.size(); sbp_start++) {
    start_node_->final_sbp_sig_id_ = sbp_start;
    start_node_out_cost[sbp_start] = start_node_->EvalOutNbhCost(node_list_id2nbh_id);
  }
  // Current Cost, Minimum Cost, Cost with original sbp
  double curr_cost = 0.0;
  double min_cost = start_node_out_cost[min_sbp_start] + end_node_out_cost[min_sbp_end]
                    + weighted_cost_[min_sbp_start][min_sbp_end];
  double original_cost = min_cost;

  for (int32_t sbp_start = 0; sbp_start < weighted_cost_.size(); sbp_start++) {
    for (int32_t sbp_end = 0; sbp_end < weighted_cost_[0].size(); sbp_end++) {
      // compute Current Cost for Neighborhood of edge
      end_node_->final_sbp_sig_id_ = sbp_end;
      curr_cost = start_node_out_cost[sbp_start] + end_node_out_cost[sbp_end]
                  + weighted_cost_[sbp_start][sbp_end];
      // Find the minimum current cost
      if (curr_cost < min_cost) {
        min_cost = curr_cost;
        min_sbp_start = sbp_start;
        min_sbp_end = sbp_end;
      }
    }
  }
  start_node_->final_sbp_sig_id_ = min_sbp_start;
  end_node_->final_sbp_sig_id_ = min_sbp_end;
  return min_cost - original_cost;
}

// Get the minimum element in Cost
double SbpEdge::GetMinWeightedCost() {
  // used the stored value if pre-computed.
  if (kMemoryRatio == memory_ratio4min_weighted_cost_ && min_weighted_cost_ >= 0) {
    return min_weighted_cost_;
  }
  // Check the size of Cost
  CHECK(weighted_cost_.size() > 0) << "Cost not initialized!" << std::endl;
  // Compute the min_cost for corresponding memory ratio
  min_weighted_cost_ = GetWeightedCost();
  for (int32_t i = 0; i < weighted_cost_.size(); i++) {
    for (int32_t j = 0; j < weighted_cost_[i].size(); j++) {
      min_weighted_cost_ = std::min(min_weighted_cost_, GetWeightedCost(i, j));
    }
  }
  // Store current the memory ratio
  memory_ratio4min_weighted_cost_ = kMemoryRatio;
  return min_weighted_cost_;
}

// Assemble copy cost
void SbpEdge::InitCopyAndMemoryCost(const std::string& ibn, bool use_sbp_collector,
                                    bool nccl_not_use_compute_stream) {
  std::vector<int64_t> consumer_nd_sbp_sig2memory;
  if (nccl_not_use_compute_stream) {
    in_memory_support_ = true;
    // Compute and store the memory for consumer
    const auto& consumer_operator = end_node_->op_node_->op();
    const auto& end_sbp_sig_list = end_node_->sbp_sig_list_;
    consumer_nd_sbp_sig2memory.resize(end_sbp_sig_list.size(), 0);
    const auto& lbi = consumer_operator.BnInOp2Lbi(ibn);
    const auto& consumer_hierarchy =
        *CHECK_JUST(consumer_operator.GetParallelDesc4BnInOp(ibn))->hierarchy();
    const auto& logical_blob_desc = start_node_->op_node_->LogicalBlobDesc4Lbi(lbi);
    HashMap<NdSbp, int64_t> consumer_nd_sbp2memory;
    for (int32_t sbp_sig_id = 0; sbp_sig_id < end_sbp_sig_list.size(); sbp_sig_id++) {
      const NdSbp& nd_sbp = end_sbp_sig_list[sbp_sig_id].bn_in_op2nd_sbp().at(ibn);
      auto it = consumer_nd_sbp2memory.find(nd_sbp);
      if (it == consumer_nd_sbp2memory.end()) {
        // This compute the memory at rank 0, the largest one.
        // We could be faster if we just compute the average memory.
        it = consumer_nd_sbp2memory
                 .insert({nd_sbp,
                          MaxByteSize4BlobDescSbp(logical_blob_desc, nd_sbp, consumer_hierarchy)})
                 .first;
      }
      consumer_nd_sbp_sig2memory[sbp_sig_id] += it->second;
    }
  }

  // In this part, we assemble the cost from nodes to nodes.
  if (start_node_->op_node_ && end_node_->op_node_) {
    OpNode* consumer = end_node_->op_node_;

    // Add copy cost for each blob
    const LogicalBlobId& lbi = consumer->op().BnInOp2Lbi(ibn);

    // Check whether lbi is transferred by this edge
    if (use_sbp_collector && !SearchLbi(lbi)) { return; }

    OpNode* producer = start_node_->op_node_;
    const std::string& producer_lbn = *CHECK_JUST(producer->op().obn4lbi(lbi));
    const ParallelDesc& producer_parallel_desc =
        *CHECK_JUST(producer->op().GetParallelDesc4BnInOp(producer_lbn));
    const ParallelDesc& consumer_parallel_desc =
        *CHECK_JUST(consumer->op().GetParallelDesc4BnInOp(ibn));

    // Need to be careful, the logical blob description should be independent to current
    // SbpParallel. Use producer or op_node?
    const BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
    const std::string& obn = *CHECK_JUST(producer->op().obn4lbi(lbi));
    // If we are deciding whether we need the wait time, then make require_same_sbp true.
    // B->S cause cudaEventSynchronize in current implementation.
    bool require_same_sbp = RequireSameSbp(consumer, ibn);
    int32_t consumer_sbp_size = end_node_->sbp_sig_list_.size();
    LazyMode::Guard enable_lazy_mode(true);

    // look through sbp signature in producer
    for (int32_t sbp_id_producer = 0; sbp_id_producer < start_node_->sbp_sig_list_.size();
         sbp_id_producer++) {
      // get sbp parallel for a logical blob in producer
      const auto& producer_sbp_bn_in_op2sbp_parallel =
          start_node_->sbp_sig_list_[sbp_id_producer].bn_in_op2nd_sbp();
      const NdSbp& sbp_producer = producer_sbp_bn_in_op2sbp_parallel.at(obn);
      auto& cost4sbp_id_producer = cost_[sbp_id_producer];

      // look through sbp signature in consumer
      for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
        // get sbp parallel for a logical blob in consumer
        const auto& consumer_sbp_bn_in_op2sbp_parallel =
            end_node_->sbp_sig_list_[sbp_id_consumer].bn_in_op2nd_sbp();
        const NdSbp& sbp_consumer = consumer_sbp_bn_in_op2sbp_parallel.at(ibn);

        // compute copy cost for a specific logical blob
        double curr_edge_cost = CHECK_JUST(ComputeCopyCostWithMiddleNodes(
            sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
            consumer_parallel_desc, require_same_sbp));
        if (curr_edge_cost < GetValidMaxCopyCost()) {
          cost4sbp_id_producer[sbp_id_consumer] +=
              CHECK_JUST(producer->op().GetOpTimeShape())->elem_cnt() * curr_edge_cost;
        } else {
          cost4sbp_id_producer[sbp_id_consumer] = curr_edge_cost;
        }
        // If enabling nccl_use_compute_stream and transfer occurs,
        // the current code would create a non-reusable register to receive data.
        if (nccl_not_use_compute_stream && curr_edge_cost > 0) {
          memory_[sbp_id_producer][sbp_id_consumer] += consumer_nd_sbp_sig2memory[sbp_id_consumer];
        }
      }
    }
  }
}

// Assemble memory cost
void SbpEdge::InitializeMemory(const HashMap<LogicalBlobId, int32_t>& lbi2id,
                               const std::vector<int32_t>& id2count,
                               const std::vector<int64_t>& producer_nd_sbp_sig2memory) {
  const auto& consumer_operator = end_node_->op_node_->op();
  const auto& end_sbp_sig_list = end_node_->sbp_sig_list_;
  std::vector<int64_t> consumer_nd_sbp_sig2memory(end_sbp_sig_list.size(), 0);
  // Compute and store the memory for consumer
  for (const auto& ibn : consumer_operator.input_bns()) {
    // Match the ibn to find the hierarchy
    const auto& lbi = consumer_operator.BnInOp2Lbi(ibn);
    if (SearchLbi(lbi) && id2count.at(lbi2id.at(lbi)) > 0) {
      const auto& consumer_hierarchy =
          *CHECK_JUST(consumer_operator.GetParallelDesc4BnInOp(ibn))->hierarchy();
      const auto& logical_blob_desc = start_node_->op_node_->LogicalBlobDesc4Lbi(lbi);
      HashMap<NdSbp, int64_t> consumer_nd_sbp2memory;
      for (int32_t sbp_sig_id = 0; sbp_sig_id < end_sbp_sig_list.size(); sbp_sig_id++) {
        const NdSbp& nd_sbp = end_sbp_sig_list[sbp_sig_id].bn_in_op2nd_sbp().at(ibn);
        auto it = consumer_nd_sbp2memory.find(nd_sbp);
        if (it == consumer_nd_sbp2memory.end()) {
          // This compute the memory at rank 0, the largest one.
          // We could be faster if we just compute the average memory.
          it = consumer_nd_sbp2memory
                   .insert({nd_sbp,
                            MaxByteSize4BlobDescSbp(logical_blob_desc, nd_sbp, consumer_hierarchy)})
                   .first;
        }
        consumer_nd_sbp_sig2memory[sbp_sig_id] += it->second;
      }
    }
  }
  // Avoid negative value for memory
  // For example, B -> S might reduce memory but we still consider 0 memory increment instead of
  // negative memory increment.
  if (*std::max_element(consumer_nd_sbp_sig2memory.begin(), consumer_nd_sbp_sig2memory.end())
      > *std::min_element(producer_nd_sbp_sig2memory.begin(), producer_nd_sbp_sig2memory.end())) {
    in_memory_support_ = true;
    memory_.resize(producer_nd_sbp_sig2memory.size());
    int32_t consumer_sbp_sig_size = consumer_nd_sbp_sig2memory.size();
    for (int32_t i = 0; i < producer_nd_sbp_sig2memory.size(); i++) {
      auto& memory_i = memory_[i];
      memory_i.resize(consumer_sbp_sig_size, 0);
      for (int32_t j = 0; j < consumer_sbp_sig_size; j++) {
        int64_t memory_difference = consumer_nd_sbp_sig2memory[j] - producer_nd_sbp_sig2memory[i];
        // Only accept positive memory change
        if (memory_difference > 0) { memory_i[j] = memory_difference; }
      }
    }
  }
}

// Set the cut ratio
double SbpEdge::GetCutRatio() const {
  int32_t num = 0;
  for (int32_t i = 0; i < weighted_cost_.size(); i++) {
    for (int32_t j = 0; j < weighted_cost_[i].size(); j++) {
      if (weighted_cost_[i][j] < GetValidMaxCopyCost()) { num++; }
    }
  }
  return double(num) / double(weighted_cost_.size() * weighted_cost_[0].size());
}

// find the cut ratio
// (#c>GetValidMaxCopyCost() in Cost)/(#c in Cost)
double SbpEdge::FindCutRatio(int32_t threshold) const {
  double cut_ratio = GetCutRatio();
  // lift the cut ratio to 1 to filter out some improper couples to avoid unlimited merging
  double n = weighted_cost_.size();
  double m = weighted_cost_[0].size();
  double num = cut_ratio * n * m;
  cut_ratio += 0.16 * (n + m) / double(threshold);
  if (num <= n * 2 || num <= m * 2 || (num <= threshold && cut_ratio < 0.51)) {
    return cut_ratio;
  } else {
    return 1.0;
  }
}

// load a logical blob
void SbpEdge::LoadLbi(const LogicalBlobId& lbi) { carry_lbis_.insert(lbi); }

// check the existence of a logical blob
bool SbpEdge::SearchLbi(const LogicalBlobId& lbi) const {
  return carry_lbis_.find(lbi) != carry_lbis_.end();
}

// unload a logical blob
void SbpEdge::UnloadLbi(const LogicalBlobId& lbi) {
  if (carry_lbis_.erase(lbi) == 0) { std::cout << "Unload an empty lbi!" << std::endl; }
}

// Not carrying any blob
bool SbpEdge::EmptyLbi() const { return carry_lbis_.empty(); }

}  // namespace auto_parallel
}  // namespace oneflow
