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

#include <algorithm>
#include <unordered_map>
#include "oneflow/core/auto_parallel/binary_set.h"
#include "oneflow/core/auto_parallel/sbp_graph.h"
#include "oneflow/core/auto_parallel/sbp_edge.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/algorithm_util.h"

namespace oneflow {
namespace auto_parallel {

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.

namespace {
static const int32_t kMinNodeInGraphForMerging = 4;
}  // anonymous namespace

// Generate a node
SbpNode* SbpGraph::GenerateNode() {
  SbpNode* this_node = new SbpNode();
  node_list_.emplace_back(this_node);
  this_node->node_list_id_ = node_list_.size() - 1;
  return this_node;
}

void SbpGraph::RemoveFromNodeList(SbpNode* this_node) {
  if (this_node->node_list_id_ < 0) { return; }
  node_list_.back()->node_list_id_ = this_node->node_list_id_;
  RemoveFrom<SbpNode*>(node_list_, this_node->node_list_id_);
  this_node->node_list_id_ = -1;
}

SbpGraph::~SbpGraph() {
  for (auto this_node : node_list_) { delete this_node; }
  node_list_.clear();
}

void SbpGraph::RandomSbpSignature(bool use_sbp_collector) const {
  for (const auto& this_node : node_list_) {
    if (this_node->sbp_sig_list_.size() > 0) {
      this_node->final_sbp_sig_id_ = rand() % this_node->sbp_sig_list_.size();
    } else {
      // It must be a proxy when this_node->sbp_sig_list_.size() == 0
      this_node->final_sbp_sig_id_ = rand() % this_node->parallel_candidates_.size();
    }
  }
};

void SbpGraph::SetDefaultSbpSig() const {
  for (const auto& this_node : node_list_) { this_node->final_sbp_sig_id_ = 0; }
};

void SbpGraph::StoreOriginMemory() {
  // We do not need to store the origin cost and memory for edges
  // Because the origin cost and memory is the current cost and memory for a bare edge.
  // For nodes, we need to do so because child elimination would attach the child cost and memory to
  // the current cost and memory.
  for (auto& this_node : node_list_) {
    this_node->origin_cost_ = this_node->cost_;
    this_node->origin_memory_ = this_node->memory_;
  }
}

double SbpGraph::ComputeCost() const {
  // Overall cost under current strategy
  double graph_cost_ = 0;
  for (const auto& this_node : node_list_) {
    int32_t this_id = this_node->final_sbp_sig_id_;

    graph_cost_ += this_node->weighted_cost_[this_id];
    for (const auto& edge_out : this_node->edges_out_) {
      graph_cost_ += edge_out->weighted_cost_[this_id][edge_out->end_node_->final_sbp_sig_id_];
    }
  }
  return graph_cost_;
}

double SbpGraph::ComputeWeightedCost() const {
  // Overall cost under current strategy
  double graph_cost_ = 0;
  for (const auto& this_node : node_list_) {
    int32_t this_id = this_node->final_sbp_sig_id_;

    graph_cost_ += this_node->weighted_cost_[this_id];
    for (const auto& edge_out : this_node->edges_out_) {
      graph_cost_ += edge_out->weighted_cost_[this_id][edge_out->end_node_->final_sbp_sig_id_];
    }
  }
  return graph_cost_;
}

// Re-compute weighted cost
void SbpGraph::ReComputeWeightedCost() {
  for (const auto& this_node : node_list_) {
    this_node->ComputeWeightedCost();
    for (const auto& edge_out : this_node->edges_out_) { edge_out->ComputeWeightedCost(); }
  }
}

int64_t SbpGraph::GetMemory() const {
  // Overall memory under current strategy
  int64_t total_memory = 0;
  for (const auto& this_node : node_list_) {
    total_memory += this_node->GetMemory();
    for (const auto& edge_out : this_node->edges_out_) { total_memory += edge_out->GetMemory(); }
  }
  return total_memory;
}

int32_t SbpGraph::NodeElimination(SbpNode* this_node) {
  if (this_node->edges_in_.size() + this_node->edges_out_.size() == 2) {
    std::vector<SbpNode*> two_nodes;
    for (const auto& one_edge : this_node->edges_in_) two_nodes.emplace_back(one_edge->start_node_);
    for (const auto& one_edge : this_node->edges_out_) two_nodes.emplace_back(one_edge->end_node_);

    // If a node is pointing to itself, could happen when shrink from a circle
    if (two_nodes[0] == two_nodes[1]) {
      int32_t elimination_number = 0;
      if (this_node->edges_out_.empty()) {
        elimination_number += EdgeElimination(two_nodes[0]);
      } else {
        elimination_number += EdgeElimination(this_node);
      }

      elimination_number += ChildElimination(this_node);
      return elimination_number;
    }

    std::vector<SbpEdge*> two_edges(this_node->edges_in_);
    two_edges.insert(two_edges.end(), this_node->edges_out_.begin(), this_node->edges_out_.end());

    int32_t edges_in_size = this_node->edges_in_.size();

    SbpEdge* e = new SbpEdge(two_nodes[0], this_node, two_nodes[1], two_edges[0], two_edges[1]);
    e->SummarizeCost();
    // check and remove the edge_in with new edge in graph
    for (int32_t i = 0; i < edges_in_size; i++) {
      CheckAndRemoveFrom<SbpEdge*>(two_nodes[i]->edges_out_, two_edges[i]);
    }
    // check and remove the edge_out with new edge in graph
    for (int32_t i = edges_in_size; i < 2; i++) {
      CheckAndRemoveFrom<SbpEdge*>(two_nodes[i]->edges_in_, two_edges[i]);
    }
    // Let e take control of edge_list_ completely by disconnecting MidNode
    e->mid_node_->edges_out_.clear();
    e->mid_node_->edges_in_.clear();

    // Insert new compound edge into graph
    two_nodes[0]->edges_out_.emplace_back(e);
    two_nodes[1]->edges_in_.emplace_back(e);

    // eliminate the node from graph by swapping with the last element and
    // popping
    RemoveFromNodeList(this_node);

    // successfully eliminate this node
    return 1;
  }
  // can not eliminate this node
  return 0;
}

int32_t SbpGraph::NodeAndEdgeEliminations() {
  // Total elimination number
  int32_t total_elimination_num = 0;
  int32_t elimination_num = 1;
  // repeat these kinds of elimination until stuck
  while (elimination_num > 0) {
    elimination_num = 0;
    for (int32_t i = node_list_.size() - 1; i >= 0; i--) {
      elimination_num += NodeElimination(node_list_[i]);
    }

    for (int32_t i = node_list_.size() - 1; i >= 0; i--) {
      elimination_num += EdgeElimination(node_list_[i]);
    }

    for (int32_t i = node_list_.size() - 1; i >= 0; i--) {
      elimination_num += ChildElimination(node_list_[i]);
    }

    if (elimination_num == 0 && node_list_.size() > 2) {
      elimination_num += PickAndMerge();
      for (int32_t i = node_list_.size() - 1; i >= 0; i--) {
        elimination_num += EdgeElimination(node_list_[i]);
      }
    }

    total_elimination_num += elimination_num;
  }

  return total_elimination_num;
}

int32_t SbpGraph::EdgeElimination(SbpNode* this_node) const {
  // Remove all edges with (start_node -> end_node) from edges_in_ of end_node
  auto RemoveFromEdgesIn = [](SbpNode* start_node, SbpNode* end_node) -> void {
    for (int32_t i = end_node->edges_in_.size() - 1; i >= 0; i--) {
      if (start_node == end_node->edges_in_[i]->start_node_) {
        RemoveFrom<SbpEdge*>(end_node->edges_in_, i);
      }
    }
  };
  auto LookForParallelEdge = [](SbpEdge*& e, SbpNode* start_node, SbpNode* end_node,
                                bool if_reverse, int32_t stop_sign) -> int32_t {
    // elimination edges with specific start node and end node in
    // start_node->edges_out_ from index stop sign to the end.
    // start_node->edges_out_[stop_sign] not included and need special treatment
    // after this process.
    int32_t elimination_num = 0;
    for (int32_t j = start_node->edges_out_.size() - 1; j > stop_sign; j--) {
      if (end_node == start_node->edges_out_[j]->end_node_) {
        if (!e) {
          if (if_reverse) {
            e = new SbpEdge(end_node, start_node);
          } else {
            e = new SbpEdge(start_node, end_node);
          }
        }
        // edge elimination
        e->edge_list_.emplace_back(start_node->edges_out_[j]);
        elimination_num++;
        RemoveFrom<SbpEdge*>(start_node->edges_out_, j);
      }
    }
    return elimination_num;
  };

  int32_t elimination_num = 0;

  for (int32_t i = 0; i < this_node->edges_out_.size(); i++) {
    SbpEdge* e = nullptr;
    // Find and delete Parallel Edges from edges_out_
    elimination_num += LookForParallelEdge(e, this_node, this_node->edges_out_[i]->end_node_,
                                           /*if_reverse=*/false, i);
    elimination_num += LookForParallelEdge(e, this_node->edges_out_[i]->end_node_, this_node,
                                           /*if_reverse=*/true, /*stop_sign=*/-1);
    if (e) {
      // Delete Parallel Edges from edges_in_
      RemoveFromEdgesIn(this_node, e->end_node_);
      RemoveFromEdgesIn(e->end_node_, this_node);
      // Add the compound edge
      e->edge_list_.emplace_back(this_node->edges_out_[i]);
      this_node->edges_out_[i] = e;
      e->SummarizeCost();
      e->end_node_->edges_in_.emplace_back(e);
    }
  }
  return elimination_num;
}

int32_t SbpGraph::ChildElimination(SbpNode* this_node) {
  if (this_node->EliminateItselfAsChild()) {
    // eliminate this node from global node list
    RemoveFromNodeList(this_node);
    // successfully eliminate this node
    return 1;
  } else {
    // can not eliminate this node
    return 0;
  }
}

// Merge two nodes
int32_t SbpGraph::NodeMerging(SbpNode* first, SbpNode* second) {
  SbpNode* new_node = new SbpNode(first, second);

  // Adjust node_list_
  RemoveFromNodeList(first);
  RemoveFromNodeList(second);

  new_node->node_list_id_ = node_list_.size();
  node_list_.emplace_back(new_node);

  return 1;
}

void SbpGraph::FinalizeSbp() const {
  for (const auto& this_node : node_list_) { this_node->FinalizeSbp(); }
}

double SbpGraph::GreedyStrategy(bool for_node) const {
  // Overall, this function should be replaced by GreedyStrategy(nbh_num);
  // Total Cost Reduce & Cost Reduce for one loop
  double total_cost_reduction = 0, cost_reduction = 0;
  for (int32_t step = node_list_.size(); step >= 0; step--) {
    cost_reduction = 0;
    for (SbpNode* this_node : node_list_) {
      // Use GreedyStrategy on Nodes if there is one node left for this
      // connected component. Otherwise, Use GreedyStrategy on Edges.
      if (for_node || this_node->edges_in_.size() + this_node->edges_out_.size() == 0) {
        cost_reduction += this_node->GreedyStrategy();
      } else {
        // GreedyStrategy on Edges.
        for (SbpEdge* this_edge : this_node->edges_out_) {
          double second_rdc = this_edge->GreedyStrategy();
          cost_reduction += second_rdc;
        }
      }
    }
    if (cost_reduction == 0) { break; }
    total_cost_reduction += cost_reduction;
  }
  return total_cost_reduction;
}

double SbpGraph::GreedyStrategy(int32_t nbh_num) const {
  // nbh_num is the maximum number of neighborhood to adjust sbp strategy in each step
  // Total Cost Reduce & Cost Reduce for one loop
  double total_cost_reduction = 0, cost_reduction = 0;
  // A global buffer to store part of the one ring neighborhood.
  std::vector<int32_t> nbh_id2node_list_id;
  // Not accept a number lower than 1
  if (nbh_num < 1) { nbh_num = 1; }
  nbh_id2node_list_id.resize(nbh_num);
  std::vector<int32_t> original_sbp_sig_id(nbh_num);
  // store all the node_list_id whose corresponding nodes will be visited
  // We can use unordered_map to do this but vector is faster
  std::vector<int32_t> pre_visit_node_list(node_list_.size() + 1);
  for (int32_t nbh_id = 0; nbh_id < node_list_.size(); nbh_id++) {
    pre_visit_node_list[nbh_id] = nbh_id;
  }
  int32_t head = 0, tail = node_list_.size();
  // whether a node_list_id is in pre_visit_node_list
  std::vector<bool> pre_visit_tags(node_list_.size(), true);
  int32_t step = 0;
  // 1 ring neighborhood buffer
  std::vector<int32_t> nbh_1ring(nbh_num);
  // 2 ring neighborhood buffer
  std::vector<int32_t> nbh_2ring;
  std::vector<bool> node_tags(node_list_.size(), false);
  std::vector<int32_t> nbh_1ring_buffer;

  while (head != tail && step < node_list_.size()) {
    auto* this_node = node_list_[pre_visit_node_list[head]];
    if (nbh_num <= 1) {
      // Greedy strategy on nodes, here we use nbh_1ring to store the nbh_id2node_list_id
      // information for reutilization
      nbh_1ring[0] = this_node->node_list_id_;
      // store the original sbp signature of the 1-ring neighborhood for comparison
      original_sbp_sig_id[0] = this_node->final_sbp_sig_id_;
      cost_reduction = NbhGreedyStrategy(nbh_1ring);
    } else {
      // Use GreedyStrategy on the one ring neighborhood of this node.
      this_node->OneRingNeighborhood(nbh_1ring);
      // store the original sbp signature of the 1-ring neighborhood for comparison
      original_sbp_sig_id.resize(nbh_1ring.size());
      for (int32_t nbh_id = 0; nbh_id < nbh_1ring.size(); nbh_id++) {
        original_sbp_sig_id[nbh_id] = node_list_[nbh_1ring[nbh_id]]->final_sbp_sig_id_;
      }
      if (nbh_1ring.size() <= nbh_num) {
        cost_reduction = NbhGreedyStrategy(nbh_1ring);
      } else {
        // Use GreedyStrategy on part of the one ring neighborhood.
        // Loop through the neighborhood. Each loop should contain the centroid.

        // Initialize part of the one ring neighborhood
        int32_t nbh_1ring_id = nbh_1ring.size() - nbh_num;
        for (int32_t nbh_id = 1; nbh_id < nbh_num; ++nbh_id) {
          nbh_id2node_list_id[nbh_id] = nbh_1ring[++nbh_1ring_id];
        }
        // loop through the one ring neighborhood
        cost_reduction = 0;
        int32_t nbh_id = 0;
        for (nbh_1ring_id = 0; nbh_1ring_id < nbh_1ring.size(); ++nbh_1ring_id) {
          nbh_id2node_list_id[nbh_id] = nbh_1ring[nbh_1ring_id];
          cost_reduction += NbhGreedyStrategy(nbh_id2node_list_id);
          // nbh_id for the next step
          if (++nbh_id >= nbh_num) { nbh_id = 1; }
        }
      }
    }
    // change of strategies
    if (cost_reduction != 0) {
      // Add neighborhood into pre-visited node list for each node with changing strategy
      for (int32_t nbh_id = 0; nbh_id < nbh_1ring.size(); nbh_id++) {
        // If changes occur
        if (original_sbp_sig_id[nbh_id] != node_list_[nbh_1ring[nbh_id]]->final_sbp_sig_id_) {
          // schedule to visit the neighborhood of that changing node
          node_list_[nbh_1ring[nbh_id]]->NRingNeighborhood(2, nbh_2ring, nbh_1ring_buffer,
                                                           node_list_, node_tags);
          for (int32_t nbh_node_list_id : nbh_2ring) {
            // Put them into the pre-visited node list
            if (!pre_visit_tags[nbh_node_list_id]) {
              pre_visit_node_list[tail] = nbh_node_list_id;
              pre_visit_tags[nbh_node_list_id] = true;
              tail++;
              if (tail == pre_visit_node_list.size()) { tail = 0; }
            }
          }
        }
      }
    }
    // Finish visiting
    pre_visit_tags[pre_visit_node_list[head]] = false;
    head++;
    if (head == pre_visit_node_list.size()) {
      head = 0;
      step++;
    }

    total_cost_reduction += cost_reduction;
  }
  return total_cost_reduction;
}

void SbpGraph::DfsAddNbhCost(std::vector<int32_t>& nbh_id2node_list_id,
                             std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                             std::vector<int32_t>& order2nbh_id, std::vector<int32_t>& nbh_id2order,
                             std::vector<double>& order2acc_min_in_nbh_cost,
                             std::vector<std::vector<double>>& out_nbh_costs,
                             std::vector<std::vector<int32_t>>& nbh_id2order2sbp_id,
                             std::vector<int32_t>& min_sbp_sig_id, double& min_cost, int32_t order,
                             double curr_cost) const {
  // We have finished visiting the neighborhood
  if (order >= nbh_id2node_list_id.size()) {
    // relative difference > 1e-12
    if (curr_cost < min_cost * kFloatDeviationMinus) {
      min_cost = curr_cost;
      for (int32_t nbh_id = 0; nbh_id < nbh_id2node_list_id.size(); nbh_id++) {
        min_sbp_sig_id[nbh_id] = node_list_[nbh_id2node_list_id[nbh_id]]->final_sbp_sig_id_;
      }
    }
    return;
  }
  // Pruning, remove all those branch with large cost
  if (curr_cost + order2acc_min_in_nbh_cost[order] >= min_cost) { return; }
  // Deep first search in the next order
  int32_t nbh_id = order2nbh_id[order];
  SbpNode* sbp_node = node_list_[nbh_id2node_list_id[nbh_id]];
  for (int32_t sbp_id : nbh_id2order2sbp_id[nbh_id]) {
    sbp_node->final_sbp_sig_id_ = sbp_id;
    DfsAddNbhCost(nbh_id2node_list_id, node_list_id2nbh_id, order2nbh_id, nbh_id2order,
                  order2acc_min_in_nbh_cost, out_nbh_costs, nbh_id2order2sbp_id, min_sbp_sig_id,
                  min_cost, order + 1,
                  curr_cost + out_nbh_costs[nbh_id][sbp_id]
                      + sbp_node->EvalInNbhCost(node_list_id2nbh_id, nbh_id2order));
  }
}

bool SbpGraph::DfsFindReasonableCost(std::vector<int32_t>& nbh_id2node_list_id,
                                     std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                                     std::vector<int32_t>& nbh_id2order, int32_t nbh_id) const {
  // We found such a strategy
  if (nbh_id == nbh_id2order.size()) { return true; }
  SbpNode* sbp_node = node_list_[nbh_id2node_list_id[nbh_id]];
  // Start from B.
  for (int32_t sbp_id = sbp_node->weighted_cost_.size() - 1; sbp_id >= 0; sbp_id--) {
    sbp_node->final_sbp_sig_id_ = sbp_id;
    // If the cost for this node is reasonable, then go to the next one
    if (sbp_node->weighted_cost_[sbp_id]
            + sbp_node->EvalInNbhCost(node_list_id2nbh_id, nbh_id2order)
        < GetValidMaxCopyCost()) {
      if (DfsFindReasonableCost(nbh_id2node_list_id, node_list_id2nbh_id, nbh_id2order,
                                nbh_id + 1)) {
        // If we found one strategy, then exist the Dfs.
        return true;
      }
    }
  }
  // Can not find a reasonable strategy with the setting for previous nodes.
  // Go back and change the previous node.
  return false;
}

// Find one strategy with finite cost for adjustment
Maybe<void> SbpGraph::Find1Strategy4Greedy() const {
  std::vector<int32_t> nbh_id2node_list_id;
  std::vector<bool> not_visited(node_list_.size(), true);
  std::vector<int32_t> nbh_1ring;
  int32_t head = 0;
  int32_t tail = 0;
  std::vector<double> node_cut_ratios(node_list_.size());
  // Initialize cut ratio for all the nodes
  for (int32_t node_list_id = 0; node_list_id < node_list_.size(); node_list_id++) {
    node_cut_ratios[node_list_id] = node_list_[node_list_id]->GetCutRatio();
  }
  // If have not visited all the nodes
  while (tail < node_list_.size()) {
    // Find the node with the minimum cut ratio
    int32_t node_with_min_cut_ratio = -1;
    double min_cut_ratio = 2.0;
    for (int32_t node_list_id = 0; node_list_id < node_list_.size(); node_list_id++) {
      if (not_visited[node_list_id]) {
        double curr_cut_ratio = node_cut_ratios[node_list_id];
        if (curr_cut_ratio < min_cut_ratio) {
          min_cut_ratio = curr_cut_ratio;
          node_with_min_cut_ratio = node_list_id;
        }
      }
    }
    // put this node into the open set
    nbh_id2node_list_id.push_back(node_with_min_cut_ratio);
    not_visited[node_with_min_cut_ratio] = false;
    tail++;
    // BFS
    while (head < tail) {
      // look for the neighborhood of the head
      int32_t node_list_id = nbh_id2node_list_id[head];
      node_list_[node_list_id]->OneRingNeighborhood(nbh_1ring);
      // sort
      std::sort(nbh_1ring.begin(), nbh_1ring.end(),
                [&](int32_t i, int32_t j) { return node_cut_ratios[i] < node_cut_ratios[j]; });
      for (int32_t curr_id : nbh_1ring) {
        if (not_visited[curr_id]) {
          nbh_id2node_list_id.push_back(curr_id);
          tail++;
          not_visited[curr_id] = false;
        }
      }
      head++;
    }
  }
  // mapping from the node_list_id to the id in the nbh_id2node_list_id
  std::unordered_map<int32_t, int32_t> node_list_id2nbh_id;
  InverseFunction<int32_t>(nbh_id2node_list_id, node_list_id2nbh_id);
  // Initial an ordinary order
  std::vector<int32_t> nbh_id2order(nbh_id2node_list_id.size());
  for (int32_t nbh_id = 0; nbh_id < nbh_id2node_list_id.size(); nbh_id++) {
    nbh_id2order[nbh_id] = nbh_id;
  }
  // Combining deep first search and pruning based on cut ratio
  CHECK(DfsFindReasonableCost(nbh_id2node_list_id, node_list_id2nbh_id, nbh_id2order, /*nbh_id=*/0))
      << "Can't find a reasonable strategy!";
  return Maybe<void>::Ok();
}

// Use brute force to search for a strategy with minimum cost for a neighborhood
double SbpGraph::NbhGreedyStrategy(std::vector<int32_t>& nbh_id2node_list_id) const {
  // number of nodes in the neighborhood
  int32_t num_nbh = nbh_id2node_list_id.size();
  // mapping from the node_list_id to the id in the nbh_id2node_list_id
  std::unordered_map<int32_t, int32_t> node_list_id2nbh_id;
  InverseFunction<int32_t>(nbh_id2node_list_id, node_list_id2nbh_id);
  // a sbp signature id set minimizing the overall cost, store the original one as default
  std::vector<int32_t> min_sbp_sig_id(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    min_sbp_sig_id[nbh_id] = node_list_[nbh_id2node_list_id[nbh_id]]->final_sbp_sig_id_;
  }

  // pre-compute and store the cost between neighborhood and outside nodes under different sbp for
  // each node within the neighborhood
  std::vector<std::vector<double>> out_nbh_costs(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    SbpNode* sbp_node = node_list_[nbh_id2node_list_id[nbh_id]];
    out_nbh_costs[nbh_id].resize(sbp_node->weighted_cost_.size());
    for (int32_t sbp_id = sbp_node->weighted_cost_.size() - 1; sbp_id >= 0; sbp_id--) {
      sbp_node->final_sbp_sig_id_ = sbp_id;
      out_nbh_costs[nbh_id][sbp_id] = sbp_node->EvalOutNbhCost(node_list_id2nbh_id);
    }
  }
  // pre-compute and store the order of the out_nbh_costs
  std::vector<std::vector<int32_t>> nbh_id2order2sbp_id(num_nbh);
  auto CompareDoubleLess = [](double a, double b) { return a < b; };
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    DecideOrder(out_nbh_costs[nbh_id], nbh_id2order2sbp_id[nbh_id], CompareDoubleLess);
  }

  // Decide the order to go through the neighborhood.
  // Should visit those nodes with a larger difference in the out cost first.
  std::vector<double> out_nbh_cost_diff(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    out_nbh_cost_diff[nbh_id] =
        *std::max_element(out_nbh_costs[nbh_id].begin(), out_nbh_costs[nbh_id].end())
        - *std::min_element(out_nbh_costs[nbh_id].begin(), out_nbh_costs[nbh_id].end());
  }
  std::vector<int32_t> order2nbh_id;
  DecideOrder(out_nbh_cost_diff, order2nbh_id, [](double a, double b) { return a > b; });
  // Find the inverse map of order
  std::vector<int32_t> nbh_id2order;
  InverseOrder(order2nbh_id, nbh_id2order);

  // Current Cost, Minimum Cost, Cost with original sbp
  double original_cost = 0;
  // Recover original sbp
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    node_list_[nbh_id2node_list_id[nbh_id]]->final_sbp_sig_id_ = min_sbp_sig_id[nbh_id];
  }
  // Compute cost with original sbp
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    SbpNode* sbp_node = node_list_[nbh_id2node_list_id[nbh_id]];
    original_cost += out_nbh_costs[nbh_id][min_sbp_sig_id[nbh_id]];
    original_cost += sbp_node->EvalInNbhCost(node_list_id2nbh_id, nbh_id2order);
  }
  double min_cost = original_cost;
  // Accumulate minimum cost from the current node to the end of the neighborhood node list.
  // The accumulated cost include the current node.
  std::vector<double> order2acc_min_in_nbh_cost(num_nbh);
  order2acc_min_in_nbh_cost[num_nbh - 1] =
      *std::min_element(out_nbh_costs[order2nbh_id[num_nbh - 1]].begin(),
                        out_nbh_costs[order2nbh_id[num_nbh - 1]].end());
  for (int32_t order = num_nbh - 2; order >= 0; order--) {
    int32_t nbh_id = order2nbh_id[order];
    order2acc_min_in_nbh_cost[order] =
        order2acc_min_in_nbh_cost[order + 1]
        + *std::min_element(out_nbh_costs[nbh_id].begin(), out_nbh_costs[nbh_id].end())
        + node_list_[nbh_id2node_list_id[nbh_id]]->EvalMinInNbhCost(node_list_id2nbh_id,
                                                                    nbh_id2order);
  }
  // Use brute force (DFS) to adjust for the best strategy in the neighborhood.
  DfsAddNbhCost(nbh_id2node_list_id, node_list_id2nbh_id, order2nbh_id, nbh_id2order,
                order2acc_min_in_nbh_cost, out_nbh_costs, nbh_id2order2sbp_id, min_sbp_sig_id,
                min_cost, /*order=*/0, /*curr_cost=*/0);
  // Use the sbp strategy with minimum cost
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    node_list_[nbh_id2node_list_id[nbh_id]]->final_sbp_sig_id_ = min_sbp_sig_id[nbh_id];
  }

  if (min_cost < original_cost) {
    // Directly return (min_cost - original_cost) might have floating point error up to 3e-16
    // For example, original_cost: 2.22507e+06, min_cost: 2.22507e+06,
    // diff: -4.65661e-10, relative diff:2.09279e-16
    // Therefore, we use a threshold to filter out such fake true detection to
    // avoid unlimited search.
    if (original_cost * kFloatDeviationMinus > min_cost) { return min_cost - original_cost; }
  }
  return 0.0;
}

// Select and Merge two nodes
int32_t SbpGraph::PickAndMerge() {
  if (node_list_.size() < kMinNodeInGraphForMerging) { return 0; }
  // Pick the one with the smallest cut ratio
  double min_cut_ratio = 1.0;
  double curr_cut_ratio = 0.0;
  SbpEdge* merging_edge = nullptr;
  for (int32_t i = 0; i < node_list_.size(); i++) {
    for (SbpEdge* edge_in : node_list_[i]->edges_in_) {
      curr_cut_ratio = edge_in->FindCutRatio(threshold_);
      if (curr_cut_ratio < min_cut_ratio) {
        min_cut_ratio = curr_cut_ratio;
        merging_edge = edge_in;
      }
    }
  }

  if (merging_edge != nullptr) {
    // Merge two nodes on the edge with the minimum cut ratio
    return NodeMerging(merging_edge->start_node_, merging_edge->end_node_);
  } else {
    // Pick the couple with the largest similar neighborhood
    std::vector<BinarySet> node_binary_sets(node_list_.size());
    for (int32_t i = 0; i < node_list_.size(); i++) {
      // Transfer edge to binary set
      node_binary_sets[i].Initialize(node_list_.size());
      node_binary_sets[i].AddEntry(i);
      for (const SbpEdge* edge_in : node_list_[i]->edges_in_) {
        node_binary_sets[i].AddEntry(edge_in->start_node_->node_list_id_);
      }
      for (const SbpEdge* edge_out : node_list_[i]->edges_out_) {
        node_binary_sets[i].AddEntry(edge_out->start_node_->node_list_id_);
      }
    }
    // Find two nodes with largest common subset
    // buffer of binary set
    BinarySet buffer_binary_set(node_list_.size());
    // Number of common edges
    int32_t max_comm_edge_num = 0, curr_comm_edge_num = 0;
    int32_t min_node_pair[2];
    // Number of Sbp Signature in merged node
    int32_t min_sbp_num = 0, curr_sbp_num = 0;
    for (int32_t i = 0; i < node_list_.size(); i++) {
      for (int32_t j = i + 1; j < node_list_.size(); j++) {
        curr_sbp_num = node_list_[i]->weighted_cost_.size() * node_list_[j]->weighted_cost_.size();
        if (curr_sbp_num <= threshold_) {
          node_binary_sets[i].IntersectionTo(node_binary_sets[j], buffer_binary_set);
          curr_comm_edge_num = buffer_binary_set.Total();
          if (curr_comm_edge_num > max_comm_edge_num
              || (curr_comm_edge_num == max_comm_edge_num && curr_sbp_num < min_sbp_num)) {
            min_node_pair[0] = i;
            min_node_pair[1] = j;
            max_comm_edge_num = curr_comm_edge_num;
            min_sbp_num = curr_sbp_num;
          }
        }
      }
    }
    if (max_comm_edge_num > 0) {
      return NodeMerging(node_list_[min_node_pair[0]], node_list_[min_node_pair[1]]);
    } else {
      return 0;
    }
  }
}

// Clip an edge, remove it from graph
void SbpGraph::ClipEdge(SbpEdge* this_edge) const {
  CheckAndRemoveFrom<SbpEdge*>(this_edge->end_node_->edges_in_, this_edge);
  CheckAndRemoveFrom<SbpEdge*>(this_edge->start_node_->edges_out_, this_edge);
  delete this_edge;
}

// Compute the minimum and maximum layer of each node in the graph
int32_t SbpGraph::ComputeLayer(
    HashMap<std::string, SbpNode*>& op_name2sbp_node,
    const HashMap<const OpNode*, HashSet<std::string>>& op_node2mutable_op_ctrl_deps) const {
  // Compute minimum layer
  for (SbpNode* this_node : node_list_) {
    this_node->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
  }
  // Find the largest minimum layer
  int32_t max_min_layer = -1;
  for (SbpNode* this_node : node_list_) {
    if (max_min_layer < this_node->min_layer_) { max_min_layer = this_node->min_layer_; }
  }
  // Compute maximum layer
  for (SbpNode* this_node : node_list_) {
    this_node->SpreadMaxLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
  }
  for (SbpNode* this_node : node_list_) { this_node->LiftMaxLayer(max_min_layer); }
  return max_min_layer;
}

// TODO: Remove the tributary layer here.
// Find the trunk of the sbp graph, then reduce the wait time for tributaries
void SbpGraph::FindTrunk(int32_t max_min_layer,
                         HashMap<std::string, SbpNode*>& op_name2sbp_node) const {
  // Summarize cost for each layer, on the trunk or tributaries
  std::vector<double> trunk_cost(max_min_layer + 1, 0);
  for (SbpNode* this_node : node_list_) {
    trunk_cost[this_node->min_layer_] += this_node->GetMinCost();
  }
  // Decide trunks
  double acc_cost = 0;
  // All the nodes with MinLayer>=trunk_end_id would be considered as trunks
  int32_t trunk_end_id = max_min_layer;
  for (int32_t layer_id = max_min_layer; layer_id >= 0; layer_id--) {
    acc_cost += trunk_cost[layer_id];
    if (acc_cost > 0.5 * wait_time_) {
      trunk_end_id = layer_id;
      break;
    }
  }
  // Find out all the nodes on the trunk.
  for (SbpNode* this_node : node_list_) {
    if (this_node->min_layer_ >= trunk_end_id) { this_node->SpreadTrunk(op_name2sbp_node); }
  }

  // Compute maximum layer for tributaries
  // Clear counter and initialize tributary layer for each sbp node
  for (SbpNode* this_node : node_list_) {
    this_node->counter_ = 0;
    this_node->DropTributaryLayer(max_min_layer);
  }
  // Count the number of consumers and downstream nodes
  for (SbpNode* this_node : node_list_) { this_node->RaiseConsumerNum(op_name2sbp_node); }
  // Compute maximum layer for tributaries
  for (SbpNode* this_node : node_list_) { this_node->SpreadTributaryLayer(op_name2sbp_node); }

  // Summarize cost for each layer on the trunk, store it to avoid subtraction of large values.
  trunk_cost.assign(max_min_layer + 1, 0);
  // tributary cost start from each min layer
  std::vector<double> tributary_cost(max_min_layer + 1, 0);
  // tributary cost would be outdated after Max Layer (before Max Layer + 1)
  std::vector<double> outdated_tributary_cost(max_min_layer + 1, 0);
  // number of operators in the trunk
  std::vector<std::vector<SbpNode*>> trunk_ops(max_min_layer + 1);

  for (SbpNode* this_node : node_list_) {
    if (this_node->on_trunk_) {
      trunk_cost[this_node->min_layer_] += this_node->GetMinCost();
      trunk_ops[this_node->min_layer_].emplace_back(this_node);
    } else {
      double curr_min_cost = this_node->GetMinCost();
      tributary_cost[this_node->min_layer_] += curr_min_cost;
      outdated_tributary_cost[this_node->tributary_layer_] += curr_min_cost;
    }
  }
  // Accumulate the cost from the consumer to the end, not including itself
  std::vector<double> acc_trunk_cost(max_min_layer + 1, 0);
  for (int32_t layer_id = max_min_layer; layer_id > 0; layer_id--) {
    acc_trunk_cost[layer_id - 1] = acc_trunk_cost[layer_id] + trunk_cost[layer_id];
  }

  // Clear counter for each sbp node
  for (SbpNode* this_node : node_list_) { this_node->counter_ = 0; }
  // Count the number of consumers and downstream nodes
  for (SbpNode* this_node : node_list_) { this_node->RaiseConsumerNum(op_name2sbp_node); }
  // Reduce the wait time for tributaries
  for (SbpNode* this_node : node_list_) {
    this_node->SpreadAvailWaitTime(trunk_cost, acc_trunk_cost, op_name2sbp_node, wait_time_);
  }

  // Reduce the wait time for trunk from the end to the begin
  double acc_tributary_cost = outdated_tributary_cost[max_min_layer];
  double used_tributary_cost = 0.0;
  double curr_wait_time = 0.0;
  for (int32_t layer_id = max_min_layer - 1; layer_id >= 0; layer_id--) {
    // Can not move it backward since we need to do this at the 0th layer.
    // At some moment, the cost haven't been used would disappear.
    if (tributary_cost[layer_id + 1] > used_tributary_cost) {
      acc_tributary_cost -= tributary_cost[layer_id + 1] - used_tributary_cost;
      used_tributary_cost = 0.0;
      if (acc_tributary_cost < 0.0) {
        // should not happen besides floating point error
        std::cout << "Caution! Current accumulated tributary cost is: " << acc_tributary_cost
                  << std::endl;
        acc_tributary_cost = 0.0;
      }
    } else {
      used_tributary_cost -= tributary_cost[layer_id + 1];
    }
    // accumulate tributary cost at this layer
    acc_tributary_cost += outdated_tributary_cost[layer_id];
    // If we have more cost in tributaries, we reduce the wait time
    // This code maintains ( acc_tributary_cost + used_tributary_cost )
    if (acc_tributary_cost > 0.0) {
      if (acc_tributary_cost > wait_time_) {
        curr_wait_time = 0.0;
        acc_tributary_cost -= wait_time_;
        used_tributary_cost += wait_time_;
      } else {
        curr_wait_time = wait_time_ - acc_tributary_cost;
        used_tributary_cost += acc_tributary_cost;
        acc_tributary_cost = 0.0;
      }
      // Reduce the wait time in the trunk
      for (SbpNode* this_node : trunk_ops[layer_id]) {
        this_node->SetTrunkWaitTime(curr_wait_time);
      }
    }
  }
}

// Set wait time
void SbpGraph::SetWaitTime(double wait_time) { wait_time_ = wait_time; }

}  // namespace auto_parallel
}  // namespace oneflow
