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

#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include "oneflow/core/auto_parallel/binary_set.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/auto_parallel/algorithm_util.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_edge.h"
#include "oneflow/core/auto_parallel/sbp_graph.h"

namespace oneflow {
namespace auto_parallel {

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.

SbpNode::SbpNode(SbpNode* first, SbpNode* second) {
  half_node_.resize(2);
  half_node_[0] = first;
  half_node_[1] = second;

  // Get the edge between first and second
  // NOTE: It must zero or one edge between them
  SbpEdge* common_edge = nullptr;
  for (int32_t k = 0; k < first->edges_in_.size(); k++) {
    if (first->edges_in_[k]->start_node_ == second) {
      // CHECK_ISNULL(edge);
      common_edge = first->edges_in_[k];
    }
  }
  for (int32_t k = 0; k < first->edges_out_.size(); k++) {
    if (first->edges_out_[k]->end_node_ == second) { common_edge = first->edges_out_[k]; }
  }

  // Find all available merged-SbpSignature(edge's cost less than threshold).
  if (common_edge) {
    double min_cost = GetMaxVal<float>();
    for (const auto& row : common_edge->cost_) {
      for (const double& c : row) min_cost = std::min(min_cost, c);
    }
    // If there is no one case can choose, we will blow up
    for (int32_t i = 0; i < first->cost_.size(); i++) {
      for (int32_t j = 0; j < second->cost_.size(); j++) {
        const double edge_cost =
            common_edge->start_node_ == first ? common_edge->cost_[i][j] : common_edge->cost_[j][i];
        if (edge_cost < GetValidMaxCopyCost()) {
          merged_sig_id2children_sig_id_.emplace_back(std::make_pair(i, j));
          cost_.emplace_back(edge_cost + first->cost_[i] + second->cost_[j]);
        }
      }
    }
    CHECK(merged_sig_id2children_sig_id_.size() > 0)
        << "0 size for merge child edge, min cost: " << min_cost;
  } else {
    for (int32_t i = 0; i < first->cost_.size(); i++) {
      for (int32_t j = 0; j < second->cost_.size(); j++) {
        merged_sig_id2children_sig_id_.emplace_back(std::make_pair(i, j));
        cost_.emplace_back(first->cost_[i] + second->cost_[j]);
      }
    }
  }

  // Initialize default sbp choice
  // If the original sbp pair does not go through, then use 0 as default.
  final_sbp_sig_id_ = 0;
  // Track the original strategy
  for (int32_t sig_id = 0; sig_id < merged_sig_id2children_sig_id_.size(); sig_id++) {
    if (merged_sig_id2children_sig_id_[sig_id].first == first->final_sbp_sig_id_
        && merged_sig_id2children_sig_id_[sig_id].second == second->final_sbp_sig_id_) {
      final_sbp_sig_id_ = sig_id;
    }
  }

  // Merge edges_in_
  edges_in_.reserve(first->edges_in_.size() + second->edges_in_.size());
  edges_in_.insert(edges_in_.end(), first->edges_in_.begin(), first->edges_in_.end());
  edges_in_.insert(edges_in_.end(), second->edges_in_.begin(), second->edges_in_.end());
  // Merge edges_out_
  edges_out_.reserve(first->edges_out_.size() + second->edges_out_.size());
  edges_out_.insert(edges_out_.end(), first->edges_out_.begin(), first->edges_out_.end());
  edges_out_.insert(edges_out_.end(), second->edges_out_.begin(), second->edges_out_.end());
  // Merge SbpEdge Cost
  for (SbpEdge*& this_edge : first->edges_in_) {
    this_edge->DuplicateCost(false, true, merged_sig_id2children_sig_id_);
    this_edge->end_node_ = this;
  }
  for (SbpEdge*& this_edge : first->edges_out_) {
    this_edge->DuplicateCost(true, true, merged_sig_id2children_sig_id_);
    this_edge->start_node_ = this;
  }
  for (SbpEdge*& this_edge : second->edges_in_) {
    this_edge->DuplicateCost(false, false, merged_sig_id2children_sig_id_);
    this_edge->end_node_ = this;
  }
  for (SbpEdge*& this_edge : second->edges_out_) {
    this_edge->DuplicateCost(true, false, merged_sig_id2children_sig_id_);
    this_edge->start_node_ = this;
  }
  // Remove edges from original nodes
  first->edges_in_.clear();
  first->edges_out_.clear();
  second->edges_in_.clear();
  second->edges_out_.clear();

  // Move edges between two nodes to each half node
  for (int32_t k = edges_out_.size() - 1; k >= 0; k--) {
    if (edges_out_[k]->end_node_ == this) {
      // Remove this edge from edges_out_ and edges_in_ and put it inside the node
      CheckAndRemoveFrom<SbpEdge*>(edges_in_, edges_out_[k]);
      first->edges_out_.emplace_back(edges_out_[k]);
      second->edges_in_.emplace_back(edges_out_[k]);
      RemoveFrom<SbpEdge*>(edges_out_, k);
    }
  }
}

SbpNode::~SbpNode() {
  for (auto& edge_out : edges_out_) { delete edge_out; }
  for (auto& child_node : children_) {
    if (child_node->edges_in_.size()) { delete child_node->edges_in_[0]; }
    delete child_node;
  }
  for (auto& half_node : half_node_) { delete half_node; }
}

void SbpNode::InitializeSbp() {
  global_sbp_sig_size_ = sbp_sig_obj_list_.size();
  sbp_sig_list_.clear();
  for (int32_t i = 0; i < sbp_sig_obj_list_.size(); i++) {
    sbp_sig_list_.emplace_back(&(sbp_sig_obj_list_[i]));
  }
  cost_.resize(sbp_sig_list_.size());
};

// Let one node point to another

void SbpNode::StartPointToEnd(SbpNode* start_node, SbpNode* end_node) {
  // generate the edge between them
  SbpEdge* e = new SbpEdge(start_node, end_node);
  start_node->edges_out_.emplace_back(e);
  end_node->edges_in_.emplace_back(e);
};

void SbpNode::PointFrom(SbpNode* start_node) { StartPointToEnd(start_node, this); };

void SbpNode::PointTo(SbpNode* end_node) { StartPointToEnd(this, end_node); };

void SbpNode::SummarizeCost() {
  if (children_.size() == child_node_sbp_sig_.size()) { return; }
  int32_t previous_children_size = child_node_sbp_sig_.size();
  child_node_sbp_sig_.resize(children_.size());
  // Only deal with new children_
  for (int32_t child = previous_children_size; child < children_.size(); child++) {
    child_node_sbp_sig_[child].resize(cost_.size());

    for (int32_t sbp_this = 0; sbp_this < cost_.size(); sbp_this++) {
      double min_cost = 0, curr_cost = 0;
      for (int32_t sbp_child = 0; sbp_child < children_[child]->cost_.size(); sbp_child++) {
        if (children_[child]->edges_in_.size()) {
          // edge in graph: father -> child
          curr_cost = children_[child]->edges_in_[0]->cost_[sbp_this][sbp_child]
                      + children_[child]->cost_[sbp_child];

        } else {
          // edge in graph: child -> father
          curr_cost = children_[child]->edges_out_[0]->cost_[sbp_child][sbp_this]
                      + children_[child]->cost_[sbp_child];
        }
        // update min_cost with fixed SbpSignature for this node and child node
        if (sbp_child == 0 || curr_cost < min_cost) {
          min_cost = curr_cost;
          child_node_sbp_sig_[child][sbp_this] = sbp_child;
        }
      }
      // Add the cost for child node to this node
      cost_[sbp_this] += min_cost;
    }
  }
}

void SbpNode::FinalizeSbp() {
  if (!half_node_.empty()) {
    // Finalize Sbp of merged nodes
    half_node_[0]->final_sbp_sig_id_ = merged_sig_id2children_sig_id_[final_sbp_sig_id_].first;
    half_node_[1]->final_sbp_sig_id_ = merged_sig_id2children_sig_id_[final_sbp_sig_id_].second;
  }

  // Finalize Sbp of children_
  for (int32_t i = 0; i < children_.size(); i++) {
    children_[i]->final_sbp_sig_id_ = child_node_sbp_sig_[i][this->final_sbp_sig_id_];
  }

  // Finalize Sbp of half_node_ Attachment
  if (!half_node_.empty()) {
    half_node_[0]->FinalizeSbp();
    half_node_[1]->FinalizeSbp();
  }

  // Finalize Sbp of edges in edges_out_
  for (const auto& edge_out : edges_out_) edge_out->FinalizeSbp();

  // Finalize Sbp again in case of the node on the other side is not finalized
  // yet. This may happen when Two side of an edge merged into two larger nodes
  // and this edge is just a sub edge.
  for (const auto& edge_in : edges_in_) edge_in->FinalizeSbp();

  // Finalize Sbp of children_ Attachment
  for (int32_t i = 0; i < children_.size(); i++) {
    children_[i]->FinalizeSbp();
    for (const auto& edge_in : children_[i]->edges_in_) edge_in->FinalizeSbp();
  }
}

double SbpNode::GreedyStrategy() {
  // Current Cost, Minimum Cost, Cost with original sbp
  double curr_cost = 0;
  double original_cost = EvalNbhCost();
  double min_cost = original_cost;
  int32_t min_sbp = final_sbp_sig_id_;
  for (int32_t sbp = 0; sbp < cost_.size(); sbp++) {
    final_sbp_sig_id_ = sbp;
    curr_cost = EvalNbhCost();
    if (curr_cost < min_cost) {
      min_cost = curr_cost;
      min_sbp = sbp;
    }
  }
  final_sbp_sig_id_ = min_sbp;
  return min_cost - original_cost;
}

double SbpNode::EvalNbhCost() const {
  // Current Cost, Minimum Cost, Cost with original sbp
  double curr_cost = cost_[final_sbp_sig_id_];
  for (SbpEdge* this_edge : edges_in_) {
    curr_cost += this_edge->cost_[this_edge->start_node_->final_sbp_sig_id_][final_sbp_sig_id_];
  }
  for (SbpEdge* this_edge : edges_out_) {
    curr_cost += this_edge->cost_[final_sbp_sig_id_][this_edge->end_node_->final_sbp_sig_id_];
  }
  return curr_cost;
}

double SbpNode::EvalOutNbhCost(
    const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id) const {
  // check if this node is in the node list
  CHECK(node_list_id_ >= 0) << "Compute out cost for a node out of the node list" << std::endl;
  // Cost with original sbp
  double curr_cost = cost_[final_sbp_sig_id_];
  for (SbpEdge* this_edge : edges_in_) {
    // if the start node is not in the neighborhood
    if (node_list_id2nbh_id.find(this_edge->start_node_->node_list_id_)
        == node_list_id2nbh_id.end()) {
      curr_cost += this_edge->cost_[this_edge->start_node_->final_sbp_sig_id_][final_sbp_sig_id_];
    }
  }
  for (SbpEdge* this_edge : edges_out_) {
    // if the end node is not in the neighborhood
    if (node_list_id2nbh_id.find(this_edge->end_node_->node_list_id_)
        == node_list_id2nbh_id.end()) {
      curr_cost += this_edge->cost_[final_sbp_sig_id_][this_edge->end_node_->final_sbp_sig_id_];
    }
  }
  return curr_cost;
}

// Compute the cost between this node and adjacent nodes with a lower order

double SbpNode::EvalInNbhCost(const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                              const std::vector<int32_t>& nbh_id2order) const {
  // check if this node is in the node list
  CHECK(node_list_id_ >= 0) << "Compute in cost for a node out of the node list";
  // check if the node is in the neighborhood
  auto this_it = node_list_id2nbh_id.find(node_list_id_);
  CHECK(this_it != node_list_id2nbh_id.end())
      << "Compute in cost for a node out of the neighborhood";
  // Compute the minimum cost between this node and adjacent nodes with a lower order
  int32_t order = nbh_id2order[this_it->second];
  double curr_cost = 0;
  for (SbpEdge* this_edge : edges_in_) {
    auto it = node_list_id2nbh_id.find(this_edge->start_node_->node_list_id_);
    // if the start node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] < order) {
      curr_cost += this_edge->cost_[this_edge->start_node_->final_sbp_sig_id_][final_sbp_sig_id_];
      // End this function and return infinity.
      if (curr_cost > GetValidMaxCopyCost()) { return GetMaxVal<float>(); }
    }
  }
  for (SbpEdge* this_edge : edges_out_) {
    auto it = node_list_id2nbh_id.find(this_edge->end_node_->node_list_id_);
    // if the end node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] < order) {
      curr_cost += this_edge->cost_[final_sbp_sig_id_][this_edge->end_node_->final_sbp_sig_id_];
      if (curr_cost > GetValidMaxCopyCost()) { return GetMaxVal<float>(); }
    }
  }
  return curr_cost;
}

double SbpNode::EvalMinInNbhCost(const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                                 const std::vector<int32_t>& nbh_id2order) const {
  // check if this node is in the node list
  CHECK(node_list_id_ >= 0) << "Compute out cost for a node out of the node list" << std::endl;
  // check if the node is in the neighborhood
  auto this_it = node_list_id2nbh_id.find(node_list_id_);
  CHECK(this_it != node_list_id2nbh_id.end())
      << "Compute out cost for a node out of the neighborhood" << std::endl;
  // Compute the minimum cost between this node and adjacent nodes with a higher order
  int32_t order = nbh_id2order[this_it->second];
  double curr_cost = 0;
  for (SbpEdge* this_edge : edges_in_) {
    auto it = node_list_id2nbh_id.find(this_edge->start_node_->node_list_id_);
    // if the start node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] > order) {
      curr_cost += this_edge->GetMinCost();
    }
  }
  for (SbpEdge* this_edge : edges_out_) {
    auto it = node_list_id2nbh_id.find(this_edge->end_node_->node_list_id_);
    // if the end node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] > order) {
      curr_cost += this_edge->GetMinCost();
    }
  }
  return curr_cost;
}

void SbpNode::OneRingNeighborhood(std::vector<int32_t>& nbh_1ring) const {
  nbh_1ring.resize(edges_in_.size() + edges_out_.size() + 1);
  int32_t nbh_id = 0;
  nbh_1ring[nbh_id] = node_list_id_;
  for (SbpEdge* this_edge : edges_in_) {
    nbh_id++;
    nbh_1ring[nbh_id] = this_edge->start_node_->node_list_id_;
  }
  for (SbpEdge* this_edge : edges_out_) {
    nbh_id++;
    nbh_1ring[nbh_id] = this_edge->end_node_->node_list_id_;
  }
}

// Get the n ring neighborhood of this node
// Pre-allocate buffer, which will be faster.

void SbpNode::NRingNeighborhood(int32_t n, std::vector<int32_t>& nbh_n_ring,
                                std::vector<int32_t>& nbh_1ring,
                                const std::vector<SbpNode*>& node_list,
                                std::vector<bool>& node_tags) const {
  // Initialize 0 ring
  if (n <= 0) { n = 0; }
  nbh_n_ring.resize(1);
  nbh_n_ring[0] = node_list_id_;
  node_tags[node_list_id_] = true;
  int32_t l = 0;
  // do ring expansion for n times
  for (int32_t i = 0; i < n; i++) {
    for (int32_t r = nbh_n_ring.size(); l < r; l++) {
      node_list[nbh_n_ring[l]]->OneRingNeighborhood(nbh_1ring);
      for (auto nbh_id : nbh_1ring) {
        if (!node_tags[nbh_id]) {
          nbh_n_ring.push_back(nbh_id);
          node_tags[nbh_id] = true;
        }
      }
    }
  }
  // Recover false for buffer
  for (auto nbh_id : nbh_n_ring) node_tags[nbh_id] = false;
}

// Get or compute the minimum layer of this node

int32_t SbpNode::GetMinLayer(const oneflow::HashMap<std::string, SbpNode*>& op_name2sbp_node,
                             const oneflow::HashMap<const OpNode*, oneflow::HashSet<std::string>>&
                                 op_node2mutable_op_ctrl_deps) {
  if (min_layer_ >= 0) { return min_layer_; }
  if (!op_node_) { return min_layer_; }
  for (SbpEdge* this_edge : edges_in_) {
    int32_t producer_min_layer =
        this_edge->start_node_->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
    if (producer_min_layer > min_layer_) { min_layer_ = producer_min_layer; }
  }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      int32_t producer_min_layer =
          it->second->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
      if (producer_min_layer > min_layer_) { min_layer_ = producer_min_layer; }
    }
  }
  if (op_node2mutable_op_ctrl_deps.find(op_node_) != op_node2mutable_op_ctrl_deps.end()) {
    for (const auto& ctrl_in_op_name : op_node2mutable_op_ctrl_deps.at(op_node_)) {
      auto it = op_name2sbp_node.find(ctrl_in_op_name);
      if (it != op_name2sbp_node.end()) {
        int32_t producer_min_layer =
            it->second->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
        if (producer_min_layer > min_layer_) { min_layer_ = producer_min_layer; }
      }
    }
  }
  return ++min_layer_;
}

// Spread the minimum layer to compute the maximum layer of producers

void SbpNode::SpreadMaxLayer(const oneflow::HashMap<std::string, SbpNode*>& op_name2sbp_node,
                             const oneflow::HashMap<const OpNode*, oneflow::HashSet<std::string>>&
                                 op_node2mutable_op_ctrl_deps) {
  if (min_layer_ <= 0) { return; }
  int32_t producer_max_lay = min_layer_ - 1;
  for (SbpEdge* this_edge : edges_in_) { this_edge->start_node_->DropMaxLayer(producer_max_lay); }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) { it->second->DropMaxLayer(producer_max_lay); }
  }
  if (op_node2mutable_op_ctrl_deps.find(op_node_) != op_node2mutable_op_ctrl_deps.end()) {
    for (const auto& ctrl_in_op_name : op_node2mutable_op_ctrl_deps.at(op_node_)) {
      auto it = op_name2sbp_node.find(ctrl_in_op_name);
      if (it != op_name2sbp_node.end()) { it->second->DropMaxLayer(producer_max_lay); }
    }
  }
}

// Drop down the maximum layer with the minimum layer form consumer

void SbpNode::DropMaxLayer(int32_t upper_bound) {
  if (upper_bound < max_layer_ || max_layer_ < 0) { max_layer_ = upper_bound; }
}
// Set max_layer_ = min_layer_ if this node does not have any consumer
// This is the end of the whole graph
// We could also set it to be the maximum of the min_layer_ in the graph. (It should be the same.)

void SbpNode::LiftMaxLayer() {
  if (max_layer_ < min_layer_) { max_layer_ = min_layer_; }
}
// Set max_layer_ = upper_bound if this node does not have any consumer

void SbpNode::LiftMaxLayer(int32_t upper_bound) {
  if (max_layer_ < min_layer_) { max_layer_ = upper_bound; }
}

// Get the minimum element in Cost

double SbpNode::GetMinCost() const {
  // Check the size of Cost
  CHECK(cost_.size() > 0) << "Cost not initialized!" << std::endl;
  // Compute the min_comp_cost
  return *std::min_element(cost_.begin(), cost_.end());
}

// Set the cut ratio

double SbpNode::GetCutRatio() const {
  double curr_cut_ratio = 1.0;
  for (auto* this_edge : edges_in_) { curr_cut_ratio *= this_edge->GetCutRatio(); }
  for (auto* this_edge : edges_out_) { curr_cut_ratio *= this_edge->GetCutRatio(); }
  return curr_cut_ratio;
}

// Judge if this node is on the trunk
// If so, judge it for its producer/upstream nodes

void SbpNode::SpreadTrunk(const oneflow::HashMap<std::string, SbpNode*>& op_name2sbp_node) {
  // Skip it if this node is already judged.
  if (on_trunk_) { return; }
  // Skip sbp proxy. This is before we have proxy.
  if (min_layer_ < 0) { return; }
  on_trunk_ = true;
  // If I am in the trunk, then all the children with (min_layer_ >= my layer id - 1) would be
  // considered as in the trunk
  for (SbpEdge* this_edge : edges_in_) {
    if (this_edge->start_node_->min_layer_ >= min_layer_ - 1) {
      this_edge->start_node_->SpreadTrunk(op_name2sbp_node);
    }
  }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end() && it->second->min_layer_ >= min_layer_ - 1) {
      it->second->SpreadTrunk(op_name2sbp_node);
    }
  }
}

// Count consumers and any downstream nodes defined by control edges

void SbpNode::RaiseConsumerNum(const oneflow::HashMap<std::string, SbpNode*>& op_name2sbp_node) {
  // Should clear it before running.
  // skip the proxy nodes and the sources
  if (min_layer_ <= 0) { return; }
  for (SbpEdge* this_edge : edges_in_) { this_edge->start_node_->counter_++; }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) { it->second->counter_++; }
  }
}

// Compute the minimal available wait time for producers or upstream nodes

void SbpNode::SpreadAvailWaitTime(const std::vector<double>& trunk_cost,
                                  const std::vector<double>& acc_trunk_cost,
                                  const oneflow::HashMap<std::string, SbpNode*>& op_name2sbp_node,
                                  double wait_time, double transfer_cost) {
  // skip the proxy nodes and the sources
  if (min_layer_ <= 0) { return; }
  // Have not finished spreading for consumers or downstream nodes or already visited.
  if (counter_) { return; }
  if (on_trunk_) {
    // Nodes on the trunk does not have any accumulate cost
    acc_trunk_cost_ = 0;
  } else {
    if (acc_trunk_cost_ < 0) {
      // Do not have any consumer or downstream node
      acc_trunk_cost_ = acc_trunk_cost[min_layer_ - 1];
    } else {
      // Add the trunk cost at this layer
      acc_trunk_cost_ += trunk_cost[min_layer_];
    }
  }

  // Reduce the wait time for edges_in_, put the rest of the trunk cost in the producers
  for (SbpEdge* this_edge : edges_in_) {
    CHECK(this_edge->wait_time_ < 0)
        << "Double assign values into wait_time_ of this edge!" << std::endl;
    SbpNode* producer = this_edge->start_node_;
    // Accumulate the cost from the start node to this node
    double curr_trunk_cost =
        acc_trunk_cost_ + acc_trunk_cost[producer->min_layer_] - acc_trunk_cost[min_layer_ - 1];
    if (curr_trunk_cost >= wait_time) {
      // Remain cost in the trunk is able to cover all the wait time
      this_edge->wait_time_ = 0.0;
      curr_trunk_cost -= wait_time;
    } else {
      // Remain cost in the trunk can only cover partial wait time
      this_edge->wait_time_ = wait_time - curr_trunk_cost;
      curr_trunk_cost = 0.0;
    }
    // Reducing non-matching edges
    // For example:
    // (1) P->S0->S0->S0->B
    // (2) p->B->B->B->B
    // We would use (2) when the tensor is relatively tiny.
    this_edge->wait_time_ += transfer_cost;
    // Do not inherit trunk cost for nodes on the trunk
    if (!producer->on_trunk_) {
      // Inherit the minimal of the trunk cost from consumers
      producer->DropAvailWaitTime(curr_trunk_cost);
    }
    producer->counter_--;
    producer->SpreadAvailWaitTime(trunk_cost, acc_trunk_cost, op_name2sbp_node, wait_time,
                                  transfer_cost);
  }
  // Put the rest the trunk cost in the upstream nodes.
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      SbpNode* producer = it->second;
      // Do not inherit trunk cost for nodes on the trunk
      if (!producer->on_trunk_) {
        // Accumulate the cost from the start node to this node
        double curr_trunk_cost =
            acc_trunk_cost_ + acc_trunk_cost[producer->min_layer_] - acc_trunk_cost[min_layer_ - 1];
        // Inherit the minimal of the trunk cost from consumers
        producer->DropAvailWaitTime(curr_trunk_cost);
      }
      producer->counter_--;
      producer->SpreadAvailWaitTime(trunk_cost, acc_trunk_cost, op_name2sbp_node, wait_time,
                                    transfer_cost);
    }
  }
  // Set counter_ to be -1, do not visit it again.
  counter_--;
}

// Drop down the available wait time with the minimum cost from downstream

void SbpNode::DropAvailWaitTime(double curr_trunk_cost) {
  if (acc_trunk_cost_ < 0.0 || acc_trunk_cost_ > curr_trunk_cost) {
    acc_trunk_cost_ = curr_trunk_cost;
  }
}

// Assemble copy cost for all the incoming edges

void SbpNode::InitializeCopyCost(bool compute_cost, bool use_sbp_collector) {
  for (SbpEdge* this_edge : edges_in_) {
    const auto* sbp_node_producer = this_edge->start_node_;
    oneflow::OpNode* producer = sbp_node_producer->op_node_;

    // skip it if proxy
    if (use_sbp_collector && !producer) { continue; }

    // look through input blobs
    for (const std::string& ibn : op_node_->op().input_bns()) {
      if (producer->op().op_name() == op_node_->SrcNode4Ibn(ibn).op().op_name()) {
        this_edge->InitializeCopyCost(ibn, compute_cost, use_sbp_collector);
      }
    }
  }
}

// Reduce and set the wait time for op in the trunk

void SbpNode::SetTrunkWaitTime(double trunk_wait_time) {
  // only reduce the wait time for operators in the trunk
  if (on_trunk_) {
    // Reduce the wait time for edges_out_
    for (SbpEdge* edge_out : edges_out_) {
      if (edge_out->wait_time_ < 0.0 || edge_out->wait_time_ > trunk_wait_time) {
        edge_out->wait_time_ = trunk_wait_time;
      }
    }
    // Might reduce it for edges_in_
  }
}

// Drop down the maximum layer with the minimum layer form consumer

void SbpNode::DropTributaryLayer(int32_t upper_bound) {
  if (upper_bound < tributary_layer_ || tributary_layer_ < 0) { tributary_layer_ = upper_bound; }
}

// Compute maximum layer for tributaries

void SbpNode::SpreadTributaryLayer(
    const oneflow::HashMap<std::string, SbpNode*>& op_name2sbp_node) {
  if (counter_ || min_layer_ <= 0) { return; }
  int32_t producer_max_lay = 0;
  if (on_trunk_) {
    producer_max_lay = min_layer_ - 1;
  } else {
    // On a tributary, the operator could be run later.
    producer_max_lay = tributary_layer_;
    // producer_max_lay = tributary_layer_ - 1;
  }
  for (SbpEdge* this_edge : edges_in_) {
    this_edge->start_node_->DropTributaryLayer(producer_max_lay);
    if (--this_edge->start_node_->counter_ == 0) {
      this_edge->start_node_->SpreadTributaryLayer(op_name2sbp_node);
    }
  }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      it->second->DropTributaryLayer(producer_max_lay);
      if (--it->second->counter_ == 0) { it->second->SpreadTributaryLayer(op_name2sbp_node); }
    }
  }
  counter_--;
}

SbpEdge* SbpNode::FindEdgeWithNode(const SbpNode* other_node) const {
  for (auto* sbp_edge : edges_in_) {
    if (sbp_edge->start_node_ == other_node) { return sbp_edge; }
  }
  for (auto* sbp_edge : edges_out_) {
    if (sbp_edge->end_node_ == other_node) { return sbp_edge; }
  }
  return nullptr;
};

// Decide to use this SbpSignature
NdSbpSignature* SbpNode::FinalSbpSignature() const {
  if (sbp_sig_list_.empty()) { return nullptr; }
  return sbp_sig_list_[final_sbp_sig_id_];
};

}  // namespace auto_parallel
}  // namespace oneflow
