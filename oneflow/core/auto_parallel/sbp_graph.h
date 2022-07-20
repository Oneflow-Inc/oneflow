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
#ifndef SBP_GRAPH_H_
#define SBP_GRAPH_H_

#include <algorithm>
#include <unordered_map>
#include "binary_set.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "sbp_edge.h"
#include "algorithm_util.h"

namespace oneflow {
namespace auto_parallel {

template<class SbpSignature>
class SbpGraph {
 public:
  // Data Structure
  // All the nodes
  std::vector<SbpNode<SbpSignature>*> NodeList;

  // Over All Cost under current strategy
  double GraphCost = 0;
  // Limitation: Merged node should not have a number of Sbp Signature greater
  // than threshold.
  int32_t Threshold = 100;
  // Overlayable wait time for copy cost, which occurs before communication between devices.
  double wait_time;
  // Uncovered wait time for copy cost.
  double transfer_cost;

  // functions
  SbpGraph();
  ~SbpGraph() {
    for (auto this_node : NodeList) { delete this_node; }
    NodeList.clear();
  }

  // Setup SbpSignature Candidates
  void AssembleSbpSignature(const std::function<int32_t()>& CalcOrderValue4SbpSig,
                            std::vector<SbpSignature*> GlobalSbpSignatureList);

  // Use our algorithm to decide SbpSignature for each op-node
  void DecideSbpSignature();

  // Randomly assign a SbpSignature strategy
  void RandomSbpSignature(bool use_sbp_collector_);
  // assign 0 to a SbpSignature strategy to avoid randomness
  void Set0SbpSignature();

  // Compute Cost for current strategy
  double ComputeCost();

  // Generate a node
  SbpNode<SbpSignature>* GenerateNode();

  // Remove a node from nodelist
  void RemoveFromNodeList(SbpNode<SbpSignature>* this_node);

  // Check and eliminate one node with only one degree-in and one degree-out
  int32_t NodeElimination(SbpNode<SbpSignature>* this_node);
  // Merge all parallel edges with given start_node_ and end_node_
  int32_t EdgeElimination(SbpNode<SbpSignature>* this_node);
  // Ckeck and eliminate one child node
  int32_t ChildElimination(SbpNode<SbpSignature>* this_node);

  // Merge all parallel edges & Check and eliminate all nodes with only one
  // degree-in and one degree-out
  int32_t NodeAndEdgeEliminations();

  // Merge two nodes
  int32_t NodeMerging(SbpNode<SbpSignature>* first, SbpNode<SbpSignature>* second);
  // Select two nodes and merge them
  int32_t PickAndMerge();

  // Finalize Sbp Cost for the whole graph
  void FinalizeSbp();

  // Use Greedy Strategy to decide Sbp for Nodes in NodeList. Should be used
  // after we have a initial strategy.
  // Set ForceNode to be true will only use GreedyStrategy on Nodes.
  double GreedyStrategy(bool ForceNode);
  // Use greedy strategy on the one ring neighborhood with the maximum number of points nbh_num.
  double GreedyStrategy(int32_t nbh_num = 4);

  // Find one strategy with finite cost for adjustment
  Maybe<void> Find1Strategy4Greedy();
  // Use brute force to search for a strategy with minimum cost for a neighborhood
  double NbhGreedyStrategy(std::vector<int32_t>& nbh_id2NodeListId);

  // Set Threshold for SbpNode Merging
  void SetThreshold(int32_t thrhld) { Threshold = thrhld; }

  // Clip an edge, remove it from graph
  // Clipping an edge will also delete the nodes and edges contained in this edge. Though not
  // sufferring from any compiling and runtime bugs, clipping an edge on a shrunk graph is not
  // recommanded. We should carefully think about it before any clipping.
  void ClipEdge(SbpEdge<SbpSignature>* this_edge);

  // Compute the minimum and maximum layer of each node in the graph
  int32_t ComputeLayer(oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node,
                       const oneflow::HashMap<const OpNode*, oneflow::HashSet<std::string>>&
                           op_node2mutable_op_ctrl_deps);

  // Find the mianstem of the sbp graph, then reduce the wait time for tributaries
  void FindMainstem(int32_t max_MinLayer,
                    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node);

  // Set wait time
  void SetWaitTime(double wait_time_);

  // Set transfer cost
  void SetTransferCost(double transfer_cost_);

 private:
  void DfsAddNbhCost(std::vector<int32_t>& nbh_id2NodeListId,
                     std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id,
                     std::vector<int32_t>& order2nbh_id, std::vector<int32_t>& nbh_id2order,
                     std::vector<double>& order2AccMinInNbhCost,
                     std::vector<std::vector<double>>& OutNbhCosts,
                     std::vector<std::vector<int32_t>>& nbh_id2order2sbp_id,
                     std::vector<int32_t>& MinSbpSignatureId, double& MinCost, int32_t order,
                     double CurrCost);

  bool DfsFindReasonableCost(std::vector<int32_t>& nbh_id2NodeListId,
                             std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id,
                             std::vector<int32_t>& nbh_id2order, int32_t nbh_id);

#ifdef DEBUG_ALGORITHM_

  // Compute Cost for current startegy with original graph
  double ComputeOriginCost();

  // Original NodeList
  std::vector<SbpNode<SbpSignature>*> OriginalNodeList;

  // get ready for Topological sorting
  void InitTopologicalSort() {
    for (const auto& this_node : NodeList) { this_node->CurrDeg = this_node->edges_in_.size(); }
  }

#endif  // DEBUG_ALGORITHM_
};

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.

// Generate a node
template<class SbpSignature>
SbpNode<SbpSignature>* SbpGraph<SbpSignature>::GenerateNode() {
  SbpNode<SbpSignature>* this_node = new SbpNode<SbpSignature>();
  NodeList.emplace_back(this_node);
  this_node->node_list_id_ = NodeList.size() - 1;
  return this_node;
}

template<class SbpSignature>
void SbpGraph<SbpSignature>::RemoveFromNodeList(SbpNode<SbpSignature>* this_node) {
  if (this_node->node_list_id_ < 0) { return; }
  NodeList.back()->node_list_id_ = this_node->node_list_id_;
  RemoveFrom<SbpNode<SbpSignature>*>(NodeList, this_node->node_list_id_);
  this_node->node_list_id_ = -1;
}

template<class SbpSignature>
SbpGraph<SbpSignature>::SbpGraph() {}

template<class SbpSignature>
void SbpGraph<SbpSignature>::AssembleSbpSignature(
    const std::function<int32_t()>& CalcOrderValue4SbpSig,
    std::vector<SbpSignature*> GlobalSbpSignatureList) {
  for (const auto& this_node : NodeList) {
    this_node->InitializeSbp(CalcOrderValue4SbpSig, GlobalSbpSignatureList);
  }
};

template<class SbpSignature>
void SbpGraph<SbpSignature>::RandomSbpSignature(bool use_sbp_collector_) {
  for (const auto& this_node : NodeList) {
    if (use_sbp_collector_) {
      if (this_node->sbp_sig_list_.size() > 0) {
        this_node->final_sbp_sig_id_ = rand() % this_node->sbp_sig_list_.size();
      } else {
        this_node->final_sbp_sig_id_ = rand() % this_node->parallel_candidates_.size();
      }
    } else {
      this_node->final_sbp_sig_id_ = rand() % this_node->sbp_sig_list_.size();
    }
  }
};

template<class SbpSignature>
void SbpGraph<SbpSignature>::Set0SbpSignature() {
  for (const auto& this_node : NodeList) { this_node->final_sbp_sig_id_ = 0; }
};

template<class SbpSignature>
double SbpGraph<SbpSignature>::ComputeCost() {
  GraphCost = 0;
  for (const auto& this_node : NodeList) {
    int32_t this_id = this_node->final_sbp_sig_id_;

    GraphCost += this_node->cost_[this_id];
    for (const auto& edge_out : this_node->edges_out_) {
      GraphCost += edge_out->cost_[this_id][edge_out->end_node_->final_sbp_sig_id_];
    }
  }
  return GraphCost;
}

template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::NodeElimination(SbpNode<SbpSignature>* this_node) {
  if (this_node->edges_in_.size() + this_node->edges_out_.size() == 2) {
    std::vector<SbpNode<SbpSignature>*> TwoNode;
    for (const auto& one_edge : this_node->edges_in_) TwoNode.emplace_back(one_edge->start_node_);
    for (const auto& one_edge : this_node->edges_out_) TwoNode.emplace_back(one_edge->end_node_);

    // If a node is pointing to itself, could happen when shrink from a circle
    if (TwoNode[0] == TwoNode[1]) {
      int32_t EliminationNumber = 0;
      if (this_node->edges_out_.empty()) {
        EliminationNumber += EdgeElimination(TwoNode[0]);
      } else {
        EliminationNumber += EdgeElimination(this_node);
      }

      EliminationNumber += ChildElimination(this_node);
      return EliminationNumber;
    }

    std::vector<SbpEdge<SbpSignature>*> TwoEdge(this_node->edges_in_);
    TwoEdge.insert(TwoEdge.end(), this_node->edges_out_.begin(), this_node->edges_out_.end());

    int32_t EdgesInSize = this_node->edges_in_.size();

    SbpEdge<SbpSignature>* e =
        new SbpEdge<SbpSignature>(TwoNode[0], this_node, TwoNode[1], TwoEdge[0], TwoEdge[1]);
    e->SummarizeCost();
    // check and remove the edge_in with new edge in graph
    for (int32_t i = 0; i < EdgesInSize; i++) {
      CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(TwoNode[i]->edges_out_, TwoEdge[i]);
    }
    // check and remove the edge_out with new edge in graph
    for (int32_t i = EdgesInSize; i < 2; i++) {
      CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(TwoNode[i]->edges_in_, TwoEdge[i]);
    }
    // Let e take control of edge_list_ completely by disconnecting MidNode
    e->mid_node_->edges_out_.clear();
    e->mid_node_->edges_in_.clear();

    // Insert new compound edge into graph
    TwoNode[0]->edges_out_.emplace_back(e);
    TwoNode[1]->edges_in_.emplace_back(e);

    // eliminate the node from graph by swaping with the last element and
    // popping
    RemoveFromNodeList(this_node);

    // successfully eliminate this node
    return 1;
  }
  // can not eliminate this node
  return 0;
}

template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::NodeAndEdgeEliminations() {
  // Total elimination number
  int32_t total_elimination_num = 0;
  int32_t elimination_num = 1;
  // repeat these kinds of elimination until stuck
  while (elimination_num > 0) {
    elimination_num = 0;
    for (int32_t i = NodeList.size() - 1; i >= 0; i--) {
      elimination_num += NodeElimination(NodeList[i]);
    }

    for (int32_t i = NodeList.size() - 1; i >= 0; i--) {
      elimination_num += EdgeElimination(NodeList[i]);
    }

    for (int32_t i = NodeList.size() - 1; i >= 0; i--) {
      elimination_num += ChildElimination(NodeList[i]);
    }

    if (elimination_num == 0 && NodeList.size() > 2) {
      elimination_num += PickAndMerge();
      for (int32_t i = NodeList.size() - 1; i >= 0; i--) {
        elimination_num += EdgeElimination(NodeList[i]);
      }
    }

    total_elimination_num += elimination_num;
  }

  return total_elimination_num;
}

template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::EdgeElimination(SbpNode<SbpSignature>* this_node) {
  // Remove all edges with (start_node -> end_node) from edges_in_ of end_node
  auto RemoveFromEdgesIn = [](SbpNode<SbpSignature>* start_node,
                              SbpNode<SbpSignature>* end_node) -> void {
    for (int32_t i = end_node->edges_in_.size() - 1; i >= 0; i--) {
      if (start_node == end_node->edges_in_[i]->start_node_) {
        RemoveFrom<SbpEdge<SbpSignature>*>(end_node->edges_in_, i);
      }
    }
  };
  auto LookForParallelEdge = [](SbpEdge<SbpSignature>*& e, SbpNode<SbpSignature>* start_node,
                                SbpNode<SbpSignature>* end_node, bool if_reverse,
                                int32_t stopsign) -> int32_t {
    // elimination edges with specific start node and end node in
    // start_node->edges_out_ from index stopsign to the end.
    // start_node->edges_out_[Stopsign] not included and need special treatment
    // after this process.
    int32_t elimination_num = 0;
    for (int32_t j = start_node->edges_out_.size() - 1; j > stopsign; j--) {
      if (end_node == start_node->edges_out_[j]->end_node_) {
        if (!e) {
          if (if_reverse) {
            e = new SbpEdge<SbpSignature>(end_node, start_node);
          } else {
            e = new SbpEdge<SbpSignature>(start_node, end_node);
          }
        }
        // edge elimination
        e->edge_list_.emplace_back(start_node->edges_out_[j]);
        elimination_num++;
        RemoveFrom<SbpEdge<SbpSignature>*>(start_node->edges_out_, j);
      }
    }
    return elimination_num;
  };

  int32_t elimination_num = 0;

  for (int32_t i = 0; i < this_node->edges_out_.size(); i++) {
    SbpEdge<SbpSignature>* e = nullptr;
    // Find and delete Parallel Edges from edges_out_
    elimination_num += LookForParallelEdge(e, this_node, this_node->edges_out_[i]->end_node_,
                                           /*if_reverse=*/false, i);
    elimination_num += LookForParallelEdge(e, this_node->edges_out_[i]->end_node_, this_node,
                                           /*if_reverse=*/true, /*stopsign=*/-1);
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

template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::ChildElimination(SbpNode<SbpSignature>* this_node) {
  if (this_node->edges_in_.size() + this_node->edges_out_.size() == 1) {
    if (this_node->edges_in_.size()) {
      // edge in graph: father -> this_node
      SbpNode<SbpSignature>* father = this_node->edges_in_[0]->start_node_;
      father->children_.emplace_back(this_node);
      CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(father->edges_out_, this_node->edges_in_[0]);
      father->SummarizeCost();
    } else {
      // edge in graph: this_node -> father
      SbpNode<SbpSignature>* father = this_node->edges_out_[0]->end_node_;
      father->children_.emplace_back(this_node);
      CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(father->edges_in_, this_node->edges_out_[0]);
      father->SummarizeCost();
    }

    // eliminate this node from global node list
    RemoveFromNodeList(this_node);

    // successfully eliminate this node
    return 1;
  }
  // can not eliminate this node
  return 0;
}

// Merge two nodes
template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::NodeMerging(SbpNode<SbpSignature>* first,
                                            SbpNode<SbpSignature>* second) {
  SbpNode<SbpSignature>* new_node = new SbpNode<SbpSignature>(first, second);

  // Adjust NodeList
  RemoveFromNodeList(first);
  RemoveFromNodeList(second);

  new_node->node_list_id_ = NodeList.size();
  NodeList.emplace_back(new_node);

  return 1;
}

template<class SbpSignature>
void SbpGraph<SbpSignature>::FinalizeSbp() {
  for (const auto& this_node : NodeList) { this_node->FinalizeSbp(); }
}

template<class SbpSignature>
double SbpGraph<SbpSignature>::GreedyStrategy(bool ForceNode) {
  // Overall, this function should be replaced by GreedyStrategy(nbh_num);
  // Total Cost Reduce & Cost Reduce for one loop
  double TtlCostRdc = 0, CostRdc;
  for (int32_t step = NodeList.size(); step >= 0; step--) {
    CostRdc = 0;
    for (SbpNode<SbpSignature>* this_node : NodeList) {
      // Use GreedyStrategy on Nodes if there is one node left for this
      // connected component. Otherwise, Use GreedyStrategy on Edges.
      if (ForceNode || this_node->edges_in_.size() + this_node->edges_out_.size() == 0) {
        CostRdc += this_node->GreedyStrategy();
      } else {
        // GreedyStrategy on Edges.
        for (SbpEdge<SbpSignature>* this_edge : this_node->edges_out_) {
          double second_rdc = this_edge->GreedyStrategy();
          CostRdc += second_rdc;
        }
      }
    }
    if (CostRdc == 0) { break; }
    TtlCostRdc += CostRdc;
  }
  return TtlCostRdc;
}

template<class SbpSignature>
double SbpGraph<SbpSignature>::GreedyStrategy(int32_t nbh_num) {
  // nbh_num is the maximum number of neighborhood to adjust sbp strategy in each step
  // Total Cost Reduce & Cost Reduce for one loop
  double TtlCostRdc = 0, CostRdc;
  // A global buffer to store part of the one ring neighborhood.
  std::vector<int32_t> nbh_id2NodeListId;
  // Not accept a number lower than 1
  if (nbh_num < 1) { nbh_num = 1; }
  nbh_id2NodeListId.resize(nbh_num);
  std::vector<int32_t> OrgSbpSignatureId(nbh_num);
  // store all the NodeListId whose corresponding nodes will be visited
  // We can use unordered_map to do this but vector is faster
  std::vector<int32_t> PreVisitNodeList(NodeList.size() + 1);
  for (int32_t nbh_id = 0; nbh_id < NodeList.size(); nbh_id++) PreVisitNodeList[nbh_id] = nbh_id;
  int32_t head = 0, tail = NodeList.size();
  // whether a NodeListId is in PreVisitNodeList
  std::vector<bool> PreVisitTags(NodeList.size(), true);
  int32_t step = 0;
  // 1 ring neighborhood buffer
  std::vector<int32_t> nbh_1ring(nbh_num);
  // 2 ring neighborhood buffer
  std::vector<int32_t> nbh_2ring;
  std::vector<bool> node_tags(NodeList.size(), false);
  std::vector<int32_t> nbh_1ring_buffer;

  while (head != tail && step < NodeList.size()) {
    auto* this_node = NodeList[PreVisitNodeList[head]];
    if (nbh_num <= 1) {
      // Greedy strategy on nodes, here we use nbh_1ring to store the nbh_id2NodeListId information
      // for reutilization
      nbh_1ring[0] = this_node->node_list_id_;
      // store the original sbp signature of the 1-ring neighborhood for comparison
      OrgSbpSignatureId[0] = this_node->final_sbp_sig_id_;
      CostRdc = NbhGreedyStrategy(nbh_1ring);
    } else {
      // Use GreedyStrategy on the one ring neighborhood of this node.
      this_node->OneRingNeighborhood(nbh_1ring);
      // store the original sbp signature of the 1-ring neighborhood for comparison
      OrgSbpSignatureId.resize(nbh_1ring.size());
      for (int32_t nbh_id = 0; nbh_id < nbh_1ring.size(); nbh_id++) {
        OrgSbpSignatureId[nbh_id] = NodeList[nbh_1ring[nbh_id]]->final_sbp_sig_id_;
      }
      if (nbh_1ring.size() <= nbh_num) {
        CostRdc = NbhGreedyStrategy(nbh_1ring);
      } else {
        // Use GreedyStrategy on part of the one ring neighborhood.
        // Loop through the neighborhood. Each loop should contain the centroid.

        // Initialize part of the one ring neighborhood
        int32_t nbh_1ring_id = nbh_1ring.size() - nbh_num;
        for (int32_t nbh_id = 1; nbh_id < nbh_num; ++nbh_id) {
          nbh_id2NodeListId[nbh_id] = nbh_1ring[++nbh_1ring_id];
        }
        // loop through the one ring neighborhood
        CostRdc = 0;
        int32_t nbh_id = 0;
        for (nbh_1ring_id = 0; nbh_1ring_id < nbh_1ring.size(); ++nbh_1ring_id) {
          nbh_id2NodeListId[nbh_id] = nbh_1ring[nbh_1ring_id];
          CostRdc += NbhGreedyStrategy(nbh_id2NodeListId);
          // nbh_id for the next step
          if (++nbh_id >= nbh_num) { nbh_id = 1; }
        }
      }
    }
    // change of strategies
    if (CostRdc != 0) {
      // Add neighborhood into pre-visited node list for each node with changing strategy
      for (int32_t nbh_id = 0; nbh_id < nbh_1ring.size(); nbh_id++) {
        // If changes occur
        if (OrgSbpSignatureId[nbh_id] != NodeList[nbh_1ring[nbh_id]]->final_sbp_sig_id_) {
          // schedule to visit the neighborhood of that changing node
          NodeList[nbh_1ring[nbh_id]]->NRingNeighborhood(2, nbh_2ring, nbh_1ring_buffer, NodeList,
                                                         node_tags);
          for (int32_t nbh_NodeListId : nbh_2ring) {
            // Put them into the pre-visited node list
            if (!PreVisitTags[nbh_NodeListId]) {
              PreVisitNodeList[tail] = nbh_NodeListId;
              PreVisitTags[nbh_NodeListId] = true;
              tail++;
              if (tail == PreVisitNodeList.size()) { tail = 0; }
            }
          }
        }
      }
    }
    // Finish visiting
    PreVisitTags[PreVisitNodeList[head]] = false;
    head++;
    if (head == PreVisitNodeList.size()) {
      head = 0;
      step++;
    }

    TtlCostRdc += CostRdc;
  }
  return TtlCostRdc;
}

template<class SbpSignature>
void SbpGraph<SbpSignature>::DfsAddNbhCost(std::vector<int32_t>& nbh_id2NodeListId,
                                           std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id,
                                           std::vector<int32_t>& order2nbh_id,
                                           std::vector<int32_t>& nbh_id2order,
                                           std::vector<double>& order2AccMinInNbhCost,
                                           std::vector<std::vector<double>>& OutNbhCosts,
                                           std::vector<std::vector<int32_t>>& nbh_id2order2sbp_id,
                                           std::vector<int32_t>& MinSbpSignatureId, double& MinCost,
                                           int32_t order, double CurrCost) {
  // We have finished visiting the neighborhood
  if (order >= nbh_id2NodeListId.size()) {
    // relative difference > 1e-12
    if (CurrCost < MinCost * 0.999999999999) {
      MinCost = CurrCost;
      for (int32_t nbh_id = 0; nbh_id < nbh_id2NodeListId.size(); nbh_id++) {
        MinSbpSignatureId[nbh_id] = NodeList[nbh_id2NodeListId[nbh_id]]->final_sbp_sig_id_;
      }
    }
    return;
  }
  // Pruning, remove all those branch with large cost
  if (CurrCost + order2AccMinInNbhCost[order] >= MinCost) { return; }
  // Deep first search in the next order
  int32_t nbh_id = order2nbh_id[order];
  SbpNode<SbpSignature>* sbp_node = NodeList[nbh_id2NodeListId[nbh_id]];
  for (int32_t sbp_id : nbh_id2order2sbp_id[nbh_id]) {
    sbp_node->final_sbp_sig_id_ = sbp_id;
    DfsAddNbhCost(nbh_id2NodeListId, NodeListId2nbh_id, order2nbh_id, nbh_id2order,
                  order2AccMinInNbhCost, OutNbhCosts, nbh_id2order2sbp_id, MinSbpSignatureId,
                  MinCost, order + 1,
                  CurrCost + OutNbhCosts[nbh_id][sbp_id]
                      + sbp_node->EvalInNbhCost(NodeListId2nbh_id, nbh_id2order));
  }
}

template<class SbpSignature>
bool SbpGraph<SbpSignature>::DfsFindReasonableCost(
    std::vector<int32_t>& nbh_id2NodeListId,
    std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id, std::vector<int32_t>& nbh_id2order,
    int32_t nbh_id) {
  // We found such a strategy
  if (nbh_id == nbh_id2order.size()) { return true; }
  SbpNode<SbpSignature>* sbp_node = NodeList[nbh_id2NodeListId[nbh_id]];
  // Start from B.
  for (int32_t sbp_id = sbp_node->cost_.size() - 1; sbp_id >= 0; sbp_id--) {
    sbp_node->final_sbp_sig_id_ = sbp_id;
    // If the cost for this node is reasonable, then go to the next one
    if (sbp_node->cost_[sbp_id] + sbp_node->EvalInNbhCost(NodeListId2nbh_id, nbh_id2order)
        < GetValidMaxCopyCost()) {
      if (DfsFindReasonableCost(nbh_id2NodeListId, NodeListId2nbh_id, nbh_id2order, nbh_id + 1)) {
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
template<class SbpSignature>
Maybe<void> SbpGraph<SbpSignature>::Find1Strategy4Greedy() {
  std::vector<int32_t> nbh_id2NodeListId;
  std::vector<bool> not_visited(NodeList.size(), true);
  std::vector<int32_t> nbh_1ring;
  int32_t head = 0;
  int32_t tail = 0;
  std::vector<double> node_cut_ratios(NodeList.size());
  // Initialize cut ratio for all the nodes
  for (int32_t NodeListId = 0; NodeListId < NodeList.size(); NodeListId++) {
    node_cut_ratios[NodeListId] = NodeList[NodeListId]->GetCutRatio();
  }
  // If have not visited all the nodes
  while (tail < NodeList.size()) {
    // Find the node with the minimum cut ratio
    int32_t node_with_min_cut_ratio = -1;
    double min_cut_ratio = 2.0;
    for (int32_t NodeListId = 0; NodeListId < NodeList.size(); NodeListId++) {
      if (not_visited[NodeListId]) {
        double curr_cut_ratio = node_cut_ratios[NodeListId];
        if (curr_cut_ratio < min_cut_ratio) {
          min_cut_ratio = curr_cut_ratio;
          node_with_min_cut_ratio = NodeListId;
        }
      }
    }
    // put this node into the open set
    nbh_id2NodeListId.push_back(node_with_min_cut_ratio);
    not_visited[node_with_min_cut_ratio] = false;
    tail++;
    // BFS
    while (head < tail) {
      // look for the neighborhood of the head
      int32_t NodeListId = nbh_id2NodeListId[head];
      NodeList[NodeListId]->OneRingNeighborhood(nbh_1ring);
      // sort
      std::sort(nbh_1ring.begin(), nbh_1ring.end(),
                [&](int32_t i, int32_t j) { return node_cut_ratios[i] < node_cut_ratios[j]; });
      for (int32_t curr_id : nbh_1ring) {
        if (not_visited[curr_id]) {
          nbh_id2NodeListId.push_back(curr_id);
          tail++;
          not_visited[curr_id] = false;
        }
      }
      head++;
    }
  }
  // mapping from the NodeListId to the id in the nbh_id2NodeListId
  std::unordered_map<int32_t, int32_t> NodeListId2nbh_id;
  InverseFunction<int32_t>(nbh_id2NodeListId, NodeListId2nbh_id);
  // Initial an ordinary order
  std::vector<int32_t> nbh_id2order(nbh_id2NodeListId.size());
  for (int32_t nbh_id = 0; nbh_id < nbh_id2NodeListId.size(); nbh_id++) {
    nbh_id2order[nbh_id] = nbh_id;
  }
  // Combining deep first search and pruning based on cut ratio
  CHECK(DfsFindReasonableCost(nbh_id2NodeListId, NodeListId2nbh_id, nbh_id2order, 0))
      << "Can't find a reasonable strateggy!";
  return Maybe<void>::Ok();
}

// Use brute force to search for a strategy with minimum cost for a neighborhood
template<class SbpSignature>
double SbpGraph<SbpSignature>::NbhGreedyStrategy(std::vector<int32_t>& nbh_id2NodeListId) {
  // number of nodes in the neighborhood
  int32_t num_nbh = nbh_id2NodeListId.size();
  // mapping from the NodeListId to the id in the nbh_id2NodeListId
  std::unordered_map<int32_t, int32_t> NodeListId2nbh_id;
  InverseFunction<int32_t>(nbh_id2NodeListId, NodeListId2nbh_id);
  // a sbp signature id set minimizing the overall cost, store the original one as default
  std::vector<int32_t> MinSbpSignatureId(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    MinSbpSignatureId[nbh_id] = NodeList[nbh_id2NodeListId[nbh_id]]->final_sbp_sig_id_;
  }

  // pre-compute and store the cost between neighborhood and outside nodes under different sbp for
  // each node within the neighborhood
  std::vector<std::vector<double>> OutNbhCosts(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    SbpNode<SbpSignature>* sbp_node = NodeList[nbh_id2NodeListId[nbh_id]];
    OutNbhCosts[nbh_id].resize(sbp_node->cost_.size());
    for (int32_t sbp_id = sbp_node->cost_.size() - 1; sbp_id >= 0; sbp_id--) {
      sbp_node->final_sbp_sig_id_ = sbp_id;
      OutNbhCosts[nbh_id][sbp_id] = sbp_node->EvalOutNbhCost(NodeListId2nbh_id);
    }
  }
  // pre-compute and store the order of the OutNbhCosts
  std::vector<std::vector<int32_t>> nbh_id2order2sbp_id(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    DecideOrder(OutNbhCosts[nbh_id], nbh_id2order2sbp_id[nbh_id],
                [](double a, double b) { return a < b; });
  }

  // Decide the order to go through the neighborhood.
  // Should visit those nodes with a larger difference in the out cost first.
  std::vector<double> OutNbhCostDiff(num_nbh);
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    OutNbhCostDiff[nbh_id] =
        *std::max_element(OutNbhCosts[nbh_id].begin(), OutNbhCosts[nbh_id].end())
        - *std::min_element(OutNbhCosts[nbh_id].begin(), OutNbhCosts[nbh_id].end());
  }
  std::vector<int32_t> order2nbh_id;
  DecideOrder(OutNbhCostDiff, order2nbh_id, [](double a, double b) { return a > b; });
  // Find the inverse map of order
  std::vector<int32_t> nbh_id2order;
  InverseOrder(order2nbh_id, nbh_id2order);

  // Current Cost, Minimum Cost, Cost with original sbp
  double MinCost, OrgCost = 0;
  // Recover original sbp
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    NodeList[nbh_id2NodeListId[nbh_id]]->final_sbp_sig_id_ = MinSbpSignatureId[nbh_id];
  }
  // Compute cost with original sbp
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    SbpNode<SbpSignature>* sbp_node = NodeList[nbh_id2NodeListId[nbh_id]];
    OrgCost += OutNbhCosts[nbh_id][MinSbpSignatureId[nbh_id]];
    OrgCost += sbp_node->EvalInNbhCost(NodeListId2nbh_id, nbh_id2order);
  }
  MinCost = OrgCost;
  // Accumulate minimum cost form the current node to the end of the neighborhood node list.
  // The accumulated cost include the current node.
  std::vector<double> order2AccMinInNbhCost(num_nbh);
  order2AccMinInNbhCost[num_nbh - 1] = *std::min_element(
      OutNbhCosts[order2nbh_id[num_nbh - 1]].begin(), OutNbhCosts[order2nbh_id[num_nbh - 1]].end());
  for (int32_t order = num_nbh - 2; order >= 0; order--) {
    int32_t nbh_id = order2nbh_id[order];
    order2AccMinInNbhCost[order] =
        order2AccMinInNbhCost[order + 1]
        + *std::min_element(OutNbhCosts[nbh_id].begin(), OutNbhCosts[nbh_id].end())
        + NodeList[nbh_id2NodeListId[nbh_id]]->EvalMinInNbhCost(NodeListId2nbh_id, nbh_id2order);
  }
  // Use brute force (DFS) to adjust for the best strategy in the neighborhood.
  DfsAddNbhCost(nbh_id2NodeListId, NodeListId2nbh_id, order2nbh_id, nbh_id2order,
                order2AccMinInNbhCost, OutNbhCosts, nbh_id2order2sbp_id, MinSbpSignatureId, MinCost,
                0, 0);
  // Use the sbp strategy with minimum cost
  for (int32_t nbh_id = 0; nbh_id < num_nbh; nbh_id++) {
    NodeList[nbh_id2NodeListId[nbh_id]]->final_sbp_sig_id_ = MinSbpSignatureId[nbh_id];
  }

  if (MinCost < OrgCost) {
    // Directly return (MinCost - OrgCost) might have floating point error up to 3e-16
    // For example, OrgCost: 2.22507e+06, MinCost: 2.22507e+06,
    // diff: -4.65661e-10, relative diff:2.09279e-16
    // Therefore, we use a threshold to filter out such fake true detection to
    // avoid unlimited search.
    if ((OrgCost - MinCost) / OrgCost > 1e-12) { return MinCost - OrgCost; }
  }
  return 0.0;
}

// Select and Merge two nodes
template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::PickAndMerge() {
  if (NodeList.size() < 4) { return 0; }
  // Pick the one with the smallest cut ratio
  double min_cut_ratio = 1.0;
  double curr_cut_ratio;
  SbpEdge<SbpSignature>* merging_edge = nullptr;
  for (int32_t i = 0; i < NodeList.size(); i++) {
    for (SbpEdge<SbpSignature>* edge_in : NodeList[i]->edges_in_) {
      curr_cut_ratio = edge_in->FindCutRatio(Threshold);
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
    std::vector<BinarySet> NodeBinarySets(NodeList.size());
    for (int32_t i = 0; i < NodeList.size(); i++) {
      // Transfer edge to binary set
      NodeBinarySets[i].Initialize(NodeList.size());
      NodeBinarySets[i].AddEntry(i);
      for (const SbpEdge<SbpSignature>* edge_in : NodeList[i]->edges_in_) {
        NodeBinarySets[i].AddEntry(edge_in->start_node_->node_list_id_);
      }
      for (const SbpEdge<SbpSignature>* edge_out : NodeList[i]->edges_out_) {
        NodeBinarySets[i].AddEntry(edge_out->start_node_->node_list_id_);
      }
    }
    // Find two nodes with largest common subset
    // buffer of binary set
    BinarySet BuffBnrSet(NodeList.size());
    // Number of common edges
    int32_t MaxCommEdgeNum = 0, CurrCommEdgeNum;
    int32_t MinNodePair[2];
    // Number of Sbp Signature in merged node
    int32_t MinSbpNum = 0, CurrSbpNum;
    for (int32_t i = 0; i < NodeList.size(); i++) {
      for (int32_t j = i + 1; j < NodeList.size(); j++) {
        CurrSbpNum = NodeList[i]->cost_.size() * NodeList[j]->cost_.size();
        if (CurrSbpNum <= Threshold) {
          NodeBinarySets[i].IntersectionTo(NodeBinarySets[j], BuffBnrSet);
          CurrCommEdgeNum = BuffBnrSet.Total();
          if (CurrCommEdgeNum > MaxCommEdgeNum
              || (CurrCommEdgeNum == MaxCommEdgeNum && CurrSbpNum < MinSbpNum)) {
            MinNodePair[0] = i;
            MinNodePair[1] = j;
            MaxCommEdgeNum = CurrCommEdgeNum;
            MinSbpNum = CurrSbpNum;
          }
        }
      }
    }
    if (MaxCommEdgeNum > 0) {
      return NodeMerging(NodeList[MinNodePair[0]], NodeList[MinNodePair[1]]);
    } else {
      return 0;
    }
  }
}

// Clip an edge, remove it from graph
template<class SbpSignature>
void SbpGraph<SbpSignature>::ClipEdge(SbpEdge<SbpSignature>* this_edge) {
  CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(this_edge->end_node_->edges_in_, this_edge);
  CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(this_edge->start_node_->edges_out_, this_edge);
  delete this_edge;
}

// Compute the minimum and maximum layer of each node in the graph
template<class SbpSignature>
int32_t SbpGraph<SbpSignature>::ComputeLayer(
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node,
    const oneflow::HashMap<const OpNode*, oneflow::HashSet<std::string>>&
        op_node2mutable_op_ctrl_deps) {
  // Compute minimum layer
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->GetMinLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
  }
  // Find the largest minimum layer
  int32_t max_MinLayer = -1;
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    if (max_MinLayer < this_node->min_layer_) { max_MinLayer = this_node->min_layer_; }
  }
  // Compute maximum layer
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->SpreadMaxLayer(op_name2sbp_node, op_node2mutable_op_ctrl_deps);
  }
  for (SbpNode<SbpSignature>* this_node : NodeList) { this_node->LiftMaxLayer(max_MinLayer); }
  return max_MinLayer;
}

// Find the mianstem of the sbp graph, then reduce the wait time for tributaries
template<class SbpSignature>
void SbpGraph<SbpSignature>::FindMainstem(
    int32_t max_MinLayer, oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node) {
  // Summerize cost for each layer, on the mainstem or tributaries
  std::vector<double> mainstem_cost(max_MinLayer + 1, 0);
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    mainstem_cost[this_node->min_layer_] += this_node->GetMinCost();
  }
  // Decide mainstems
  double acc_cost = 0;
  // All the nodes with MinLayer>=mainstem_end_id would be considerd as mainstems
  int32_t mainstem_end_id = max_MinLayer;
  for (int32_t layer_id = max_MinLayer; layer_id >= 0; layer_id--) {
    acc_cost += mainstem_cost[layer_id];
    if (acc_cost > 0.5 * wait_time) {
      mainstem_end_id = layer_id;
      break;
    }
  }
  // Find out all the nodes on the mainstem.
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    if (this_node->min_layer_ >= mainstem_end_id) { this_node->SpreadMainstem(op_name2sbp_node); }
  }

  // Compute maximum layer for tributaries
  // Clear counter and initialize tributary layer for each sbp node
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->counter_ = 0;
    this_node->DropTributaryLayer(max_MinLayer);
  }
  // Count the number of consumers and downstream nodes
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->RaiseConsumerNum(op_name2sbp_node);
  }
  // Compute maximum layer for tributaries
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->SpreadTributaryLayer(op_name2sbp_node);
  }

  // Summerize cost for each layer on the mainstem, store it to avoid substraction of large values.
  mainstem_cost.assign(max_MinLayer + 1, 0);
  // tributary cost start from each min layer
  std::vector<double> tributary_cost(max_MinLayer + 1, 0);
  // tributary cost would be outdated after Max Layer (before Max Layer + 1)
  std::vector<double> outdated_tributary_cost(max_MinLayer + 1, 0);
  // number of operators in the mainstem
  std::vector<std::vector<SbpNode<SbpSignature>*>> mainstem_ops(max_MinLayer + 1);

  for (SbpNode<SbpSignature>* this_node : NodeList) {
    if (this_node->on_mainstem_) {
      mainstem_cost[this_node->min_layer_] += this_node->GetMinCost();
      mainstem_ops[this_node->min_layer_].emplace_back(this_node);
    } else {
      double curr_min_cost = this_node->GetMinCost();
      tributary_cost[this_node->min_layer_] += curr_min_cost;
      outdated_tributary_cost[this_node->tributary_layer_] += curr_min_cost;
    }
  }
  // Accumulate the cost from the consumer to the end, not including itself
  std::vector<double> acc_mainstem_cost(max_MinLayer + 1, 0);
  for (int32_t layer_id = max_MinLayer; layer_id > 0; layer_id--) {
    acc_mainstem_cost[layer_id - 1] = acc_mainstem_cost[layer_id] + mainstem_cost[layer_id];
  }

  // Clear counter for each sbp node
  for (SbpNode<SbpSignature>* this_node : NodeList) { this_node->counter_ = 0; }
  // Count the number of consumers and downstream nodes
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->RaiseConsumerNum(op_name2sbp_node);
  }
  // Reduce the wait time for tributaries
  for (SbpNode<SbpSignature>* this_node : NodeList) {
    this_node->SpreadAvailWaitTime(mainstem_cost, acc_mainstem_cost, op_name2sbp_node, wait_time,
                                   transfer_cost);
  }

  // Reduce the wait time for mainstem from the end to the begining
  double acc_tributary_cost = outdated_tributary_cost[max_MinLayer];
  double used_tributary_cost = 0.0;
  double curr_wait_time;
  for (int32_t layer_id = max_MinLayer - 1; layer_id >= 0; layer_id--) {
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
    // This code maintains ( acc_triburary_cost + used_tributary_cost )
    if (acc_tributary_cost > 0.0) {
      if (acc_tributary_cost > wait_time) {
        curr_wait_time = transfer_cost;
        acc_tributary_cost -= wait_time;
        used_tributary_cost += wait_time;
      } else {
        curr_wait_time = transfer_cost + wait_time - acc_tributary_cost;
        used_tributary_cost += acc_tributary_cost;
        acc_tributary_cost = 0.0;
      }
      // Reduce the wait time in the mainstem
      for (SbpNode<SbpSignature>* this_node : mainstem_ops[layer_id]) {
        this_node->SetMainstemWaitTime(curr_wait_time);
      }
    }
  }
}

// Set wait time
template<class SbpSignature>
void SbpGraph<SbpSignature>::SetWaitTime(double wait_time_) {
  wait_time = wait_time_;
}

// Set transfer cost
template<class SbpSignature>
void SbpGraph<SbpSignature>::SetTransferCost(double transfer_cost_) {
  transfer_cost = transfer_cost_;
}

#ifdef DEBUG_ALGORITHM_

template<class SbpSignature>
double SbpGraph<SbpSignature>::ComputeOriginCost() {
  double GraphCost = 0;
  for (SbpNode<SbpSignature>* this_node : OriginalNodeList) {
    int32_t this_id = this_node->final_sbp_sig_id_;
    GraphCost += this_node->OriginCost[this_id];
    for (int32_t j = 0; j < this_node->OriginCostOut.size(); j++) {
      GraphCost += this_node->OriginCostOut[j][this_id][this_node->NodesOut[j]->final_sbp_sig_id_];
    }
  }
  return GraphCost;
}

#endif  // DEBUG_ALGORITHM_

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // SBP_GRAPH_H_
