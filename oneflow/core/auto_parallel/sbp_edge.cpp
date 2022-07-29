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

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.

SbpEdge::SbpEdge(SbpNode* start_node, SbpNode* mid_node, SbpNode* end_node, SbpEdge* first_edge,
                 SbpEdge* second_edge)
    : start_node_(start_node), mid_node_(mid_node), end_node_(end_node) {
  edge_list_.emplace_back(first_edge);
  edge_list_.emplace_back(second_edge);
};

void SbpEdge::SummarizeCost() {
  if (mid_node_) {
    cost_.resize(start_node_->cost_.size());
    mid_node_sbp_sig_.resize(start_node_->cost_.size());
    int32_t end_node_sbp_size = end_node_->cost_.size();
    int32_t mid_node_sbp_size = mid_node_->cost_.size();
    for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
      cost_[sbp_start].resize(end_node_sbp_size);
      mid_node_sbp_sig_[sbp_start].resize(end_node_sbp_size);
      for (int32_t sbp_end = 0; sbp_end < end_node_sbp_size; sbp_end++) {
        for (int32_t sbp_mid = 0; sbp_mid < mid_node_sbp_size; sbp_mid++) {
          // Add middle node cost
          double temp_cost = mid_node_->cost_[sbp_mid];
          // Add first edge cost
          if (edge_list_[0]->start_node_ == start_node_) {
            temp_cost += edge_list_[0]->cost_[sbp_start][sbp_mid];
          } else {
            temp_cost += edge_list_[0]->cost_[sbp_mid][sbp_start];
          }
          // Add second edge cost
          if (edge_list_[1]->end_node_ == end_node_) {
            temp_cost += edge_list_[1]->cost_[sbp_mid][sbp_end];
          } else {
            temp_cost += edge_list_[1]->cost_[sbp_end][sbp_mid];
          }

          // Compare and look for the minimum cost
          if (sbp_mid == 0) {
            cost_[sbp_start][sbp_end] = temp_cost;
            mid_node_sbp_sig_[sbp_start][sbp_end] = sbp_mid;
          } else if (temp_cost < cost_[sbp_start][sbp_end]) {
            cost_[sbp_start][sbp_end] = temp_cost;
            mid_node_sbp_sig_[sbp_start][sbp_end] = sbp_mid;
          }
        }
      }
    }
  } else {
    cost_.resize(start_node_->cost_.size());
    int32_t end_node_sbp_size = end_node_->cost_.size();
    for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
      cost_[sbp_start].resize(end_node_sbp_size);
      for (int32_t sbp_end = 0; sbp_end < end_node_sbp_size; sbp_end++) {
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

void SbpEdge::DuplicateCost(
    bool merged_node_is_start_node, bool duplicating_first_node,
    const std::vector<std::pair<int32_t, int32_t>>& merged_sig_id2children_sig_id) {
  const int32_t num_sig = merged_sig_id2children_sig_id.size();
  std::vector<std::vector<double>> temp_cost;
  std::vector<std::vector<int32_t>> temp_mid_node_sbp_sig;
  if (merged_node_is_start_node) {
    temp_cost.resize(num_sig);
    if (mid_node_) { temp_mid_node_sbp_sig.resize(num_sig); }
    for (int32_t i = 0; i < num_sig; i++) {
      const int32_t sig_idx = duplicating_first_node ? merged_sig_id2children_sig_id[i].first
                                                     : merged_sig_id2children_sig_id[i].second;
      temp_cost[i] = cost_[sig_idx];
      if (mid_node_) { temp_mid_node_sbp_sig[i] = mid_node_sbp_sig_[sig_idx]; }
    }
  } else {
    const int32_t num_start_sig = cost_.size();
    temp_cost.resize(num_start_sig);
    if (mid_node_) { temp_mid_node_sbp_sig.resize(num_start_sig); }
    for (int32_t i = 0; i < num_start_sig; i++) {
      temp_cost[i].resize(num_sig);
      if (mid_node_) { temp_mid_node_sbp_sig[i].resize(num_sig); }
      for (int32_t j = 0; j < num_sig; j++) {
        const int32_t sig_idx = duplicating_first_node ? merged_sig_id2children_sig_id[j].first
                                                       : merged_sig_id2children_sig_id[j].second;
        temp_cost[i][j] = cost_[i][sig_idx];
        if (mid_node_) { temp_mid_node_sbp_sig[i][j] = mid_node_sbp_sig_[i][sig_idx]; }
      }
    }
  }

  cost_ = temp_cost;
  if (mid_node_) { mid_node_sbp_sig_ = temp_mid_node_sbp_sig; }
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
  std::vector<double> end_node_out_cost(end_node_->cost_.size());
  for (int32_t sbp_end = 0; sbp_end < cost_[0].size(); sbp_end++) {
    end_node_->final_sbp_sig_id_ = sbp_end;
    end_node_out_cost[sbp_end] = end_node_->EvalOutNbhCost(node_list_id2nbh_id);
  }
  // pre-compute and store the current cost between start_node_ and outside.
  std::vector<double> start_node_out_cost(start_node_->cost_.size());
  for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
    start_node_->final_sbp_sig_id_ = sbp_start;
    start_node_out_cost[sbp_start] = start_node_->EvalOutNbhCost(node_list_id2nbh_id);
  }
  // Current Cost, Minimum Cost, Cost with original sbp
  double curr_cost = 0.0;
  double min_cost = start_node_out_cost[min_sbp_start] + end_node_out_cost[min_sbp_end]
                    + cost_[min_sbp_start][min_sbp_end];
  double original_cost = min_cost;

  for (int32_t sbp_start = 0; sbp_start < cost_.size(); sbp_start++) {
    for (int32_t sbp_end = 0; sbp_end < cost_[0].size(); sbp_end++) {
      // compute Current Cost for Neighborhood of edge
      end_node_->final_sbp_sig_id_ = sbp_end;
      curr_cost =
          start_node_out_cost[sbp_start] + end_node_out_cost[sbp_end] + cost_[sbp_start][sbp_end];
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

double SbpEdge::GetMinCost() {
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

double SbpEdge::GetMaxCost() const {
  // used the stored value if pre-computed.
  // if (max_cost >= 0) return max_cost;
  // Check the size of Cost
  CHECK(cost_.size() > 0) << "Cost not initialized!" << std::endl;
  // Compute the max_cost
  double max_cost = -1.0;
  for (int32_t i = 0; i < cost_.size(); i++) {
    for (int32_t j = 0; j < cost_[i].size(); j++) {
      if (cost_[i][j] < GetValidMaxCopyCost() && cost_[i][j] > max_cost) { max_cost = cost_[i][j]; }
    }
  }
  return max_cost;
}

// Assemble copy cost

void SbpEdge::InitializeCopyCost(const std::string& ibn, bool compute_cost,
                                 bool use_sbp_collector) {
  // In this part, we assemble the cost from nodes to nodes.
  if (start_node_->op_node_ && end_node_->op_node_) {
    oneflow::OpNode* consumer = end_node_->op_node_;

    // Add copy cost for each blob
    const oneflow::LogicalBlobId& lbi = consumer->op().BnInOp2Lbi(ibn);

    // Check whether lbi is transferred by this edge
    if (use_sbp_collector && compute_cost && !SearchLbi(lbi)) { return; }

    oneflow::OpNode* producer = start_node_->op_node_;
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
    int32_t consumer_sbp_size = end_node_->sbp_sig_list_.size();
    LazyMode::Guard enable_lazy_mode(true);

    // look through sbp signature in producer
    for (int32_t sbp_id_producer = 0; sbp_id_producer < start_node_->sbp_sig_list_.size();
         sbp_id_producer++) {
      // get sbp parallel for a logical blob in producer
      const auto producer_sbp_bn_in_op2sbp_parallel =
          start_node_->sbp_sig_list_[sbp_id_producer]->bn_in_op2nd_sbp();
      const NdSbp& sbp_producer = producer_sbp_bn_in_op2sbp_parallel.at(obn);

      // look through sbp signature in consumer
      for (int32_t sbp_id_consumer = 0; sbp_id_consumer < consumer_sbp_size; sbp_id_consumer++) {
        // get sbp parallel for a logical blob in consumer
        const auto consumer_sbp_bn_in_op2sbp_parallel =
            end_node_->sbp_sig_list_[sbp_id_consumer]->bn_in_op2nd_sbp();
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

double SbpEdge::GetCutRatio() const {
  int32_t num = 0;
  for (int32_t i = 0; i < cost_.size(); i++) {
    for (int32_t j = 0; j < cost_[i].size(); j++) {
      if (cost_[i][j] < GetValidMaxCopyCost()) { num++; }
    }
  }
  return double(num) / double(cost_.size() * cost_[0].size());
}

// find the cut ratio
// (#c>GetValidMaxCopyCost() in Cost)/(#c in Cost)

double SbpEdge::FindCutRatio(int32_t threshold) const {
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
