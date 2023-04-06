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
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace auto_parallel {

// In dynamic programming, we can not minimize a vector (copy cost, memory cost)
// Instead, we minimize the weighted sum of the vector, copy cost + kMemoryRatio * memory cost
extern double kMemoryRatio;

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
    in_memory_support_ =
        first->in_memory_support_ || second->in_memory_support_ || common_edge->in_memory_support_;
    // If there is no one case can choose, we will blow up
    for (int32_t i = 0; i < first->weighted_cost_.size(); i++) {
      for (int32_t j = 0; j < second->weighted_cost_.size(); j++) {
        const double edge_weighted_cost = common_edge->start_node_ == first
                                              ? common_edge->weighted_cost_[i][j]
                                              : common_edge->weighted_cost_[j][i];
        if (edge_weighted_cost < GetValidMaxCopyCost()) {
          merged_sig_id2half_sig_id_.emplace_back(std::make_pair(i, j));
          if (in_memory_support_) {
            memory_.push_back((common_edge->start_node_ == first ? common_edge->GetMemory(i, j)
                                                                 : common_edge->GetMemory(j, i))
                              + first->GetMemory(i) + second->GetMemory(j));
          }
          weighted_cost_.emplace_back(edge_weighted_cost + first->weighted_cost_[i]
                                      + second->weighted_cost_[j]);
        }
      }
    }
    CHECK(merged_sig_id2half_sig_id_.size() > 0)
        << "0 size for merge two half nodes with common edge!";
  } else {
    in_memory_support_ = first->in_memory_support_ || second->in_memory_support_;
    for (int32_t i = 0; i < first->weighted_cost_.size(); i++) {
      for (int32_t j = 0; j < second->weighted_cost_.size(); j++) {
        merged_sig_id2half_sig_id_.emplace_back(std::make_pair(i, j));
        if (in_memory_support_) { memory_.push_back(first->GetMemory(i) + second->GetMemory(j)); }
        weighted_cost_.emplace_back(first->weighted_cost_[i] + second->weighted_cost_[j]);
      }
    }
  }

  // Initialize default sbp choice
  // If the original sbp pair does not go through, then use 0 as default.
  final_sbp_sig_id_ = 0;
  // Track the original strategy
  for (int32_t sig_id = 0; sig_id < merged_sig_id2half_sig_id_.size(); sig_id++) {
    if (merged_sig_id2half_sig_id_[sig_id].first == first->final_sbp_sig_id_
        && merged_sig_id2half_sig_id_[sig_id].second == second->final_sbp_sig_id_) {
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
    this_edge->DuplicateCost(false, true, merged_sig_id2half_sig_id_);
    this_edge->end_node_ = this;
  }
  for (SbpEdge*& this_edge : first->edges_out_) {
    this_edge->DuplicateCost(true, true, merged_sig_id2half_sig_id_);
    this_edge->start_node_ = this;
  }
  for (SbpEdge*& this_edge : second->edges_in_) {
    this_edge->DuplicateCost(false, false, merged_sig_id2half_sig_id_);
    this_edge->end_node_ = this;
  }
  for (SbpEdge*& this_edge : second->edges_out_) {
    this_edge->DuplicateCost(true, false, merged_sig_id2half_sig_id_);
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
  global_sbp_sig_size_ = sbp_sig_list_.size();
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
  in_memory_support_ =
      in_memory_support_
      || std::any_of(children_.begin() + previous_children_size, children_.end(),
                     [](SbpNode* sbp_node) { return sbp_node->in_memory_support_; });
  if (in_memory_support_) { memory_.resize(weighted_cost_.size(), 0); }
  // Buffer
  int64_t min_memory_cost = 0, memory_cost = 0;
  double min_weighted_sum = 0.0, weighted_sum = 0.0;
  int32_t min_sbp_child = 0;
  // Only deal with new children_
  for (int32_t child = previous_children_size; child < children_.size(); child++) {
    child_node_sbp_sig_[child].resize(weighted_cost_.size());

    for (int32_t sbp_this = 0; sbp_this < weighted_cost_.size(); sbp_this++) {
      SbpNode* child_node = children_[child];
      for (int32_t sbp_child = 0; sbp_child < child_node->weighted_cost_.size(); sbp_child++) {
        if (child_node->edges_in_.size()) {
          // edge in graph: father -> child
          memory_cost = child_node->edges_in_[0]->GetMemory(sbp_this, sbp_child)
                        + child_node->GetMemory(sbp_child);
          weighted_sum = child_node->edges_in_[0]->weighted_cost_[sbp_this][sbp_child]
                         + child_node->weighted_cost_[sbp_child];
        } else {
          // edge in graph: child -> father
          memory_cost = child_node->edges_out_[0]->GetMemory(sbp_child, sbp_this)
                        + child_node->GetMemory(sbp_child);
          weighted_sum = child_node->edges_out_[0]->weighted_cost_[sbp_child][sbp_this]
                         + child_node->weighted_cost_[sbp_child];
        }
        // update min_cost with fixed SbpSignature for this node and child node
        if (sbp_child == 0 || weighted_sum < min_weighted_sum) {
          min_memory_cost = memory_cost;
          min_weighted_sum = weighted_sum;
          min_sbp_child = sbp_child;
        }
      }
      child_node_sbp_sig_[child][sbp_this] = min_sbp_child;
      // Add the cost for child node to this node
      if (in_memory_support_) { memory_[sbp_this] += min_memory_cost; }
      weighted_cost_[sbp_this] += min_weighted_sum;
    }
  }
}

bool SbpNode::EliminateItselfAsChild() {
  if (edges_in_.size() + edges_out_.size() == 1) {
    if (edges_in_.size()) {
      // edge in graph: father -> this_node
      SbpNode* father = edges_in_[0]->start_node_;
      father->children_.emplace_back(this);
      CheckAndRemoveFrom<SbpEdge*>(father->edges_out_, edges_in_[0]);
      father->SummarizeCost();
    } else {
      // edge in graph: this_node -> father
      SbpNode* father = edges_out_[0]->end_node_;
      father->children_.emplace_back(this);
      CheckAndRemoveFrom<SbpEdge*>(father->edges_in_, edges_out_[0]);
      father->SummarizeCost();
    }
    // successfully eliminate this node
    return true;
  }
  // can not eliminate this node
  return false;
}

// Compute the weighted sum of the time and memory cost
void SbpNode::ComputeWeightedCost() {
  if (half_node_.empty()) {
    // If this node is not generated from merging, it should have original cost
    // weighted_cost_ = cost_;
    weighted_cost_ = origin_cost_;
    memory_ = origin_memory_;
    if (in_memory_support_) {
      for (int32_t sbp_id = 0; sbp_id < origin_memory_.size(); sbp_id++) {
        weighted_cost_[sbp_id] += kMemoryRatio * origin_memory_[sbp_id];
      }
    }
  } else {
    half_node_[0]->ComputeWeightedCost();
    half_node_[1]->ComputeWeightedCost();
    // The edge between two half nodes
    SbpEdge* edge_found = nullptr;
    if (!half_node_[0]->edges_in_.empty()) {
      edge_found = half_node_[0]->edges_in_[0];
    } else if (!half_node_[0]->edges_out_.empty()) {
      edge_found = half_node_[0]->edges_out_[0];
    }
    if (edge_found != nullptr) { edge_found->ComputeWeightedCost(); }
    // Compute the weighted cost form half nodes
    for (int32_t merged_sig_id = 0; merged_sig_id < merged_sig_id2half_sig_id_.size();
         merged_sig_id++) {
      const auto& pair = merged_sig_id2half_sig_id_[merged_sig_id];
      if (in_memory_support_) {
        memory_[merged_sig_id] =
            half_node_[0]->GetMemory(pair.first) + half_node_[1]->GetMemory(pair.second);
      }
      weighted_cost_[merged_sig_id] =
          half_node_[0]->weighted_cost_[pair.first] + half_node_[1]->weighted_cost_[pair.second];
      if (edge_found != nullptr) {
        // The dimension of weighted cost has been expand for the found edge.
        // Both the dimension of weighted_cost_ is merged_sig_id2half_sig_id_.size().
        // The start node and end node is changed to this for the found edge.
        if (in_memory_support_) {
          memory_[merged_sig_id] += edge_found->GetMemory(merged_sig_id, merged_sig_id);
        }
        weighted_cost_[merged_sig_id] += edge_found->weighted_cost_[merged_sig_id][merged_sig_id];
      }
    }
  }
  // Compute the weighted cost for children
  for (auto& child_node : children_) {
    child_node->ComputeWeightedCost();
    for (auto& in_edge : child_node->edges_in_) { in_edge->ComputeWeightedCost(); }
    for (auto* out_edge : child_node->edges_out_) { out_edge->ComputeWeightedCost(); }
  }
  // Compute the weighted cost from children
  child_node_sbp_sig_.clear();
  SummarizeCost();
}

// Generate the relationship between this merged node and its components
void SbpNode::GenerateComponentRelationship() {
  // Do nothing if not merged node or already generated
  if (half_node_.empty() || !component2merged_sig_id2component_sig_id_.empty()) { return; }
  // Add the map for two half nodes
  auto& first_merged2component_id = component2merged_sig_id2component_sig_id_[half_node_[0]];
  auto& second_merged2component_id = component2merged_sig_id2component_sig_id_[half_node_[1]];
  int32_t total_sbp_num = weighted_cost_.size();
  first_merged2component_id.resize(total_sbp_num);
  second_merged2component_id.resize(total_sbp_num);
  for (int32_t i = 0; i < total_sbp_num; i++) {
    first_merged2component_id[i] = merged_sig_id2half_sig_id_[i].first;
    second_merged2component_id[i] = merged_sig_id2half_sig_id_[i].second;
  }
  // Add the map for the half of the half nodes
  for (int32_t i = 0; i < 2; i++) {
    half_node_[i]->GenerateComponentRelationship();
    auto& merged2half_id = component2merged_sig_id2component_sig_id_[half_node_[i]];
    for (auto& pair : half_node_[i]->component2merged_sig_id2component_sig_id_) {
      auto& merged2component_id = component2merged_sig_id2component_sig_id_[pair.first];
      merged2component_id.resize(total_sbp_num);
      auto& half2component_id = pair.second;
      for (int32_t merged_id = 0; merged_id < total_sbp_num; merged_id++) {
        merged2component_id[merged_id] = half2component_id[merged2half_id[merged_id]];
      }
    }
  }
}

void SbpNode::FinalizeSbp() {
  if (!half_node_.empty()) {
    // Finalize Sbp of merged nodes
    half_node_[0]->final_sbp_sig_id_ = merged_sig_id2half_sig_id_[final_sbp_sig_id_].first;
    half_node_[1]->final_sbp_sig_id_ = merged_sig_id2half_sig_id_[final_sbp_sig_id_].second;
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
  for (const auto& edge_out : edges_out_) { edge_out->FinalizeSbp(); }

  // Finalize Sbp again in case of the node on the other side is not finalized
  // yet. This may happen when Two side of an edge merged into two larger nodes
  // and this edge is just a sub edge.
  for (const auto& edge_in : edges_in_) { edge_in->FinalizeSbp(); }

  // Finalize Sbp of children_ Attachment
  for (int32_t i = 0; i < children_.size(); i++) {
    children_[i]->FinalizeSbp();
    for (const auto& edge_in : children_[i]->edges_in_) { edge_in->FinalizeSbp(); }
  }
}

double SbpNode::GreedyStrategy() {
  // Current Cost, Minimum Cost, Cost with original sbp
  double curr_cost = 0;
  double original_cost = EvalNbhCost();
  double min_cost = original_cost;
  int32_t min_sbp = final_sbp_sig_id_;
  for (int32_t sbp = 0; sbp < weighted_cost_.size(); sbp++) {
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
  double curr_cost = GetWeightedCost();
  for (SbpEdge* this_edge : edges_in_) { curr_cost += this_edge->GetWeightedCost(); }
  for (SbpEdge* this_edge : edges_out_) { curr_cost += this_edge->GetWeightedCost(); }
  return curr_cost;
}

double SbpNode::EvalOutNbhCost(
    const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id) const {
  // check if this node is in the node list
  CHECK(node_list_id_ >= 0) << "Compute out cost for a node out of the node list" << std::endl;
  // Cost with original sbp
  double curr_cost = GetWeightedCost();
  for (SbpEdge* this_edge : edges_in_) {
    // if the start node is not in the neighborhood
    if (node_list_id2nbh_id.find(this_edge->start_node_->node_list_id_)
        == node_list_id2nbh_id.end()) {
      curr_cost += this_edge->GetWeightedCost();
    }
  }
  for (SbpEdge* this_edge : edges_out_) {
    // if the end node is not in the neighborhood
    if (node_list_id2nbh_id.find(this_edge->end_node_->node_list_id_)
        == node_list_id2nbh_id.end()) {
      curr_cost += this_edge->GetWeightedCost();
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
  const auto& this_it = node_list_id2nbh_id.find(node_list_id_);
  CHECK(this_it != node_list_id2nbh_id.end())
      << "Compute in cost for a node out of the neighborhood";
  // Compute the minimum cost between this node and adjacent nodes with a lower order
  int32_t order = nbh_id2order[this_it->second];
  double curr_cost = 0;
  for (SbpEdge* this_edge : edges_in_) {
    const auto& it = node_list_id2nbh_id.find(this_edge->start_node_->node_list_id_);
    // if the start node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] < order) {
      curr_cost += this_edge->GetWeightedCost();
      // End this function and return infinity.
      if (curr_cost > GetValidMaxCopyCost()) { return GetMaxVal<float>(); }
    }
  }
  for (SbpEdge* this_edge : edges_out_) {
    const auto& it = node_list_id2nbh_id.find(this_edge->end_node_->node_list_id_);
    // if the end node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] < order) {
      curr_cost += this_edge->GetWeightedCost();
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
  const auto& this_it = node_list_id2nbh_id.find(node_list_id_);
  CHECK(this_it != node_list_id2nbh_id.end())
      << "Compute out cost for a node out of the neighborhood" << std::endl;
  // Compute the minimum cost between this node and adjacent nodes with a higher order
  int32_t order = nbh_id2order[this_it->second];
  double curr_cost = 0;
  for (SbpEdge* this_edge : edges_in_) {
    const auto& it = node_list_id2nbh_id.find(this_edge->start_node_->node_list_id_);
    // if the start node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] > order) {
      curr_cost += this_edge->GetMinWeightedCost();
    }
  }
  for (SbpEdge* this_edge : edges_out_) {
    const auto& it = node_list_id2nbh_id.find(this_edge->end_node_->node_list_id_);
    // if the end node is in the neighborhood
    if (it != node_list_id2nbh_id.end() && nbh_id2order[it->second] > order) {
      curr_cost += this_edge->GetMinWeightedCost();
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
  for (auto nbh_id : nbh_n_ring) { node_tags[nbh_id] = false; }
}

// Get or compute the minimum layer of this node
int32_t SbpNode::GetMinLayer(
    const HashMap<std::string, SbpNode*>& op_name2sbp_node,
    const HashMap<const OpNode*, HashSet<std::string>>& op_node2mutable_op_ctrl_deps) {
  if (min_layer_ >= 0) { return min_layer_; }
  if (!op_node_) { return min_layer_; }
  for (SbpEdge* this_edge : edges_in_) {
    int32_t producer_min_layer =
        this_edge->start_node_->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
    if (producer_min_layer > min_layer_) { min_layer_ = producer_min_layer; }
  }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      int32_t producer_min_layer =
          it->second->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
      if (producer_min_layer > min_layer_) { min_layer_ = producer_min_layer; }
    }
  }
  if (op_node2mutable_op_ctrl_deps.find(op_node_) != op_node2mutable_op_ctrl_deps.end()) {
    for (const auto& ctrl_in_op_name : op_node2mutable_op_ctrl_deps.at(op_node_)) {
      const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
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
void SbpNode::SpreadMaxLayer(
    const HashMap<std::string, SbpNode*>& op_name2sbp_node,
    const HashMap<const OpNode*, HashSet<std::string>>& op_node2mutable_op_ctrl_deps) {
  if (min_layer_ <= 0) { return; }
  int32_t producer_max_lay = min_layer_ - 1;
  for (SbpEdge* this_edge : edges_in_) { this_edge->start_node_->DropMaxLayer(producer_max_lay); }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) { it->second->DropMaxLayer(producer_max_lay); }
  }
  if (op_node2mutable_op_ctrl_deps.find(op_node_) != op_node2mutable_op_ctrl_deps.end()) {
    for (const auto& ctrl_in_op_name : op_node2mutable_op_ctrl_deps.at(op_node_)) {
      const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
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
  // Can not use weighted cost here since this function is used for find trunk.
  // We have not initialize weighted cost at this moment
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
void SbpNode::SpreadTrunk(const HashMap<std::string, SbpNode*>& op_name2sbp_node) {
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
    const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end() && it->second->min_layer_ >= min_layer_ - 1) {
      it->second->SpreadTrunk(op_name2sbp_node);
    }
  }
}

// Count consumers and any downstream nodes defined by control edges
void SbpNode::RaiseConsumerNum(const HashMap<std::string, SbpNode*>& op_name2sbp_node) {
  // Should clear it before running.
  // skip the proxy nodes and the sources
  if (min_layer_ <= 0) { return; }
  for (SbpEdge* this_edge : edges_in_) { this_edge->start_node_->counter_++; }
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) { it->second->counter_++; }
  }
}

// Compute the minimal available wait time for producers or upstream nodes
void SbpNode::SpreadAvailWaitTime(const std::vector<double>& trunk_cost,
                                  const std::vector<double>& acc_trunk_cost,
                                  const HashMap<std::string, SbpNode*>& op_name2sbp_node,
                                  double wait_time) {
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
    // Do not inherit trunk cost for nodes on the trunk
    if (!producer->on_trunk_) {
      // Inherit the minimal of the trunk cost from consumers
      producer->DropAvailWaitTime(curr_trunk_cost);
    }
    producer->counter_--;
    producer->SpreadAvailWaitTime(trunk_cost, acc_trunk_cost, op_name2sbp_node, wait_time);
  }
  // Put the rest the trunk cost in the upstream nodes.
  for (const auto& ctrl_in_op_name : op_node_->op().op_conf().ctrl_in_op_name()) {
    const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
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
      producer->SpreadAvailWaitTime(trunk_cost, acc_trunk_cost, op_name2sbp_node, wait_time);
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

// Assemble copy cost and partial memory cost for all the incoming edges
void SbpNode::InitCopyAndMemoryCost(bool use_sbp_collector, bool nccl_not_use_compute_stream) {
  for (SbpEdge* this_edge : edges_in_) {
    const auto* sbp_node_producer = this_edge->start_node_;
    OpNode* producer = sbp_node_producer->op_node_;

    // skip it if proxy
    if (use_sbp_collector && !producer) { continue; }
    // look through input blobs
    for (const std::string& ibn : op_node_->op().input_bns()) {
      if (producer->op().op_name() == op_node_->SrcNode4Ibn(ibn).op().op_name()) {
        this_edge->InitCopyAndMemoryCost(ibn, use_sbp_collector, nccl_not_use_compute_stream);
      }
    }
    // Add Wait time
    for (auto& cost_row : this_edge->cost_) {
      for (auto& cost_value : cost_row) {
        // If transferring between devices, we need to add wait time.
        if (cost_value > 0.0) { cost_value += this_edge->wait_time_; }
      }
    }
  }
}

// Assemble memory cost
void SbpNode::InitializeMemory(bool is_reusable, const HashMap<LogicalBlobId, int32_t>& lbi2id,
                               const std::vector<int32_t>& id2count, bool nccl_use_compute_stream) {
  const auto& curr_operator = op_node_->op();
  // An edge should not be initialized twice
  // During each initialization, we are computing sum(memory of consumer) - sum(memory of producer)
  // This is why we need to pre-store memory of producer
  HashMap<SbpEdge*, std::vector<int64_t>> sbp_edge2nd_sbp_sig2memory;
  for (const auto& obn : curr_operator.output_bns()) {
    const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(obn);
    // Fixed memory or in the support of the reusable memory
    if (!is_reusable || id2count.at(lbi2id.at(lbi)) > 0) {
      // If not in support, memory_ would be empty.
      in_memory_support_ = true;
      memory_.resize(sbp_sig_list_.size(), 0);
      const auto& logical_blob_desc = op_node_->LogicalBlobDesc4Lbi(lbi);
      const auto& hierarchy = *CHECK_JUST(curr_operator.GetParallelDesc4BnInOp(obn))->hierarchy();
      // There are some operators with a fixed sbp for some blobs, such as conv.
      // {in: S0, kernel: B, out: S0}
      // {in: B, kernel: B, out: B}
      // The blob kernel have the same sbp for different signatures.
      // We pre-store the results for the same sbp while accessing the same blobs.
      HashMap<NdSbp, int64_t> nd_sbp2memory;
      SbpEdge* edge_contain_lbi = nullptr;
      for (const auto& edge_out : edges_out_) {
        if (edge_out->SearchLbi(lbi)) { edge_contain_lbi = edge_out; }
      }
      // There exist some lbi which does not have a consumer
      // At this moment edge_contain_lbi == nullptr
      auto& nd_sbp_sig2memory = sbp_edge2nd_sbp_sig2memory[edge_contain_lbi];
      nd_sbp_sig2memory.resize(sbp_sig_list_.size(), 0);
      for (int32_t sbp_sig_id = 0; sbp_sig_id < sbp_sig_list_.size(); sbp_sig_id++) {
        const NdSbp& nd_sbp = sbp_sig_list_[sbp_sig_id].bn_in_op2nd_sbp().at(obn);
        auto it = nd_sbp2memory.find(nd_sbp);
        if (it == nd_sbp2memory.end()) {
          // This compute the memory at rank 0, the largest one.
          // We could be faster if we just compute the average memory.
          it = nd_sbp2memory
                   .insert({nd_sbp, MaxByteSize4BlobDescSbp(logical_blob_desc, nd_sbp, hierarchy)})
                   .first;
        }
        memory_[sbp_sig_id] += it->second;
        nd_sbp_sig2memory[sbp_sig_id] += it->second;
      }
    }
  }
  // Even after the correction in the memory of edges, the relative error still have 0.73%.
  if (nccl_use_compute_stream && in_memory_support_ && is_reusable) {
    for (const auto& pair : sbp_edge2nd_sbp_sig2memory) {
      // Init memory for each out-going edge
      pair.first->InitializeMemory(lbi2id, id2count, pair.second);
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
void SbpNode::SpreadTributaryLayer(const HashMap<std::string, SbpNode*>& op_name2sbp_node) {
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
    const auto& it = op_name2sbp_node.find(ctrl_in_op_name);
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
const NdSbpSignature& SbpNode::FinalSbpSignature() const {
  CHECK(!sbp_sig_list_.empty()) << "Asking for sbp signature for an empty node";
  return sbp_sig_list_[final_sbp_sig_id_];
};

int32_t SbpNode::GetComponentSbpId(int32_t merged_id, SbpNode* component_node) const {
  if (this == component_node) { return merged_id; }
  CHECK(!component2merged_sig_id2component_sig_id_.empty())
      << "Check the component before initialization!" << std::endl;
  return component2merged_sig_id2component_sig_id_.at(component_node).at(merged_id);
}

// Judge if sbp_node is a port of the current node
bool SbpNode::IsComponent(SbpNode* sbp_node) const {
  if (this == sbp_node) { return true; }
  // If IsComponent() is call before we initialize component2merged_sig_id2component_sig_id_,
  // we would also return false.
  // Please do not call GenerateComponentRelationship() at here.
  // Please see SbpEdge::SummarizeCost() for more details.
  return component2merged_sig_id2component_sig_id_.find(sbp_node)
         != component2merged_sig_id2component_sig_id_.end();
}

}  // namespace auto_parallel
}  // namespace oneflow
