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
#ifndef SBP_NODE_H_
#define SBP_NODE_H_

#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include "binary_set.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/graph/op_graph.h"
#include "algorithm_util.h"

#define ms_1 1e11
#define us_1 1e8
#define s_1 1e14
#define cut_cost 3e38

namespace oneflow {
namespace auto_parallel {

template<class SbpSignature>
class SbpEdge;

template<class SbpSignature>
class SbpNode {
 public:
  // Data Structure

  // compound edge in
  std::vector<SbpEdge<SbpSignature>*> EdgesIn;
  // compound edge out
  std::vector<SbpEdge<SbpSignature>*> EdgesOut;
  // Identity, use it to distinguish itself from node set
  int32_t id = -1;

  // We should use Sbp-signature for edge with lowest OrderValue
  std::vector<int32_t> OrderValue;
  // Lowest OrderValue
  int32_t LowOrderValue = -1;
  // Available SbpSignature pointer for this node
  std::vector<SbpSignature*> SbpSignatureList;
  // Available SbpSignature object for this node
  std::vector<SbpSignature> SbpSignatureObjList;
  // Global SbpSignature List Size
  int32_t GlobalSbpSigSize = -1;
  // Decide to use SbpSignature with this id
  int32_t FinalSbpSignatureId;
  // Location in NodeList
  int32_t NodeListId = -1;

  // Child node list
  std::vector<SbpNode<SbpSignature>*> Children;
  // SbpSignature for each child node when using specific SbpSignature for this
  // node Its dimension is Number of Child Nodes * Number of Available
  // SbpSignatures for this node
  std::vector<std::vector<int32_t>> ChildNodeSbpSig;

  // Merge two nodes into this compound node
  std::vector<SbpNode<SbpSignature>*> HalfNode;
  // We should delete those merged-signatures which has very large cost for speed up
  // New SbpSignatureList index map to each HalfNode's sig_index
  std::vector<std::pair<int32_t, int32_t>> MergedSigId2ChildrenSigId;

  // Cost[sbp] is Computation Cost when using SbpSignatureList[sbp]
  std::vector<double> Cost;

  std::vector<BinarySet> ParallelCandidates;

  oneflow::OpNode* op_node = nullptr;

  // We devide the sbp graph into multiple layers.
  // MinLayer is the minimum layer number to run this op as soon as possible.
  // MaxLayer is the maximum layer number without slowing down the whole process of the graph.
  // producer.MaxLayer < this_node.MinLayer <= this_node.MaxLayer < consumer.MinLayer
  int32_t MinLayer = -1, MaxLayer = -1;
  // Maximum layer in tributaries
  int32_t TributaryLayer = -1;
  // Whether we are on the mainstem
  bool IfMainstem = false;
  // A counter buffer for topological traversal or something else
  int32_t counter = 0;
  // Accumulate mainstem cost from consumer to the end
  double AccMainstemCost = -1.0;

#ifdef DEBUG_ALGORITHM_

  // original edge out
  std::vector<SbpNode*> NodesOut;
  // original cost for edge out
  std::vector<std::vector<std::vector<double>>> OriginCostOut;
  // original edge in
  std::vector<SbpNode*> NodesIn;
  // Original Cost
  std::vector<double> OriginCost;

  // Current Degree is a tag used for Topological ordering
  int32_t CurrDeg;
#endif  // DEBUG_ALGORITHM_

  // default constructor
  SbpNode() : FinalSbpSignatureId(0) {}

  // This constructor is to merge two node into one
  SbpNode(SbpNode<SbpSignature>* first, SbpNode<SbpSignature>* second);

  ~SbpNode() {
    for (auto& edge_out : EdgesOut) { delete edge_out; }
    for (auto& childnode : Children) {
      if (childnode->EdgesIn.size()) { delete childnode->EdgesIn[0]; }
      delete childnode;
    }
    for (auto& half_node : HalfNode) { delete half_node; }
  }

  // another node point to this node
  void PointFrom(SbpNode<SbpSignature>* start_node);
  // this node point to another node
  void PointTo(SbpNode<SbpSignature>* end_node);

  // initialize the OrderValue and Find the lowest one
  void FindLowOrderValue(const std::function<int32_t()>& CalcOrderValue4SbpSig);
  // Initialize SbpSignature
  void InitializeSbp(const std::function<int32_t()>& CalcOrderValue4SbpSig,
                     std::vector<SbpSignature*> GlobalSbpSignatureList);
  // Initialize SbpSignature from Signature Objects
  void InitializeSbp();
  // Compute Computation Cost
  void ComputeCost(
      const std::function<double(SbpNode<SbpSignature>*, SbpSignature*)>& SbpComputationCost);
  // Decide to use this SbpSignature
  SbpSignature* FinalSbpSignature() const {
    if (SbpSignatureList.empty()) { return NULL; }
    return SbpSignatureList[FinalSbpSignatureId];
  };

  // Recompute Computation Cost after adding child nodes in it
  void SummarizeCost();
  // Determine Final SbpSignature for attachment of this node
  void FinalizeSbp();
  // Use Greedy Strategy to pick the sbp signature with minimum cost for this
  // node You should have an initial strategy before running this
  double GreedyStrategy();
  // Evaluate summery of cost in 1-ring neighborhood.
  double EvalNbhCost();
  // Evaluate summery of cost between neighborhood and outside nodes
  double EvalOutNbhCost(std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id);
  // Evaluate summery of cost within neighborhood
  // We only accumulate the edge cost with a lower order.
  double EvalInNbhCost(std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id,
                       std::vector<int32_t>& nbh_id2order);
  // Evaluate summery of cost within neighborhood
  // We only accumulate the minimum edge cost with a higher order.
  double EvalMinInNbhCost(std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id,
                          std::vector<int32_t>& nbh_id2order);
  // Get the one ring neighborhood of this node, which is itself and all the adjacent nodes.
  void OneRingNeighborhood(std::vector<int32_t>& nbh_1ring);
  // Get the n ring neighborhood of this node
  // Pre-allocate buffer, which will be faster.
  void NRingNeighborhood(int32_t n, std::vector<int32_t>& nbh_nring,
                         std::vector<int32_t>& nbh_1ring,
                         std::vector<SbpNode<SbpSignature>*>& NodeList,
                         std::vector<bool>& node_tags);

  // Detect and spread overlaps for EdgesOut.
  void DetectSpreadOverlap(double max_1_comp_cost, double max_2_comp_cost, int32_t max_1_id,
                           double min_ratio);
  // Detect and spread overlaps for sbp proxy.
  void DetectSpreadOverlap(double overlap_ratio);

  // Get or compute the minimum layer of this node
  int32_t GetMinLayer(oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node);
  // Spread the minimum layer to compute the maximum layer of producers
  void SpreadMaxLayer(oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node);
  // Drop down the maximum layer with the minimum layer form consumer
  void DropMaxLayer(int32_t upper_bound);
  // Set MaxLayer = MinLayer if this node does not have any consumer
  void LiftMaxLayer();
  // Set MaxLayer = upper_bound if this node does not have any consumer
  void LiftMaxLayer(int32_t upper_bound);
  // Compute maximum layer for tributaries
  void SpreadTributaryLayer(
      oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node);
  // Drop down the tributary layer
  void DropTributaryLayer(int32_t upper_bound);

  // Get the minimum element in Cost
  double GetMinCost();
  // get the cut ratio
  double GetCutRatio();

  // Judge if this node is on the mainstem
  // If so, judge it for its producer/upstream nodes
  void SpreadMainstem(oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node);
  // Count consumers and any downstream nodes defined by control edges
  // for producers or upstream nodes
  void RaiseConsumerNum(oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node);
  // Compute the minimal available wait time for producers or upstream nodes
  void SpreadAvailWaitTime(std::vector<double>& mainstem_cost,
                           std::vector<double>& acc_mainstem_cost,
                           oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node,
                           double wait_time, double transfer_cost);
  // Drop down the available wait time with the minimum cost from downstreams
  void DropAvailWaitTime(double curr_mainstem_cost);
  // Reduce and set the wait time for op in the mainstem
  void SetMainstemWaitTime(double mainstem_wait_time);

  // Assemble copy cost for all the incoming edges
  void InitializeCopyCost(bool compute_cost, bool use_sbp_collector_);

};  // class SbpNode

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.

template<class SbpSignature>
SbpNode<SbpSignature>::SbpNode(SbpNode<SbpSignature>* first, SbpNode<SbpSignature>* second) {
  HalfNode.resize(2);
  HalfNode[0] = first;
  HalfNode[1] = second;

  // Get the edge between first and second
  // NOTE: It must zero or one edge between them
  SbpEdge<SbpSignature>* common_edge = nullptr;
  for (int32_t k = 0; k < first->EdgesIn.size(); k++) {
    if (first->EdgesIn[k]->StartNode == second) {
      // CHECK_ISNULL(edge);
      common_edge = first->EdgesIn[k];
    }
  }
  for (int32_t k = 0; k < first->EdgesOut.size(); k++) {
    if (first->EdgesOut[k]->EndNode == second) {
      // CHECK_ISNULL(edge);
      common_edge = first->EdgesOut[k];
    }
  }

  // Find all available merged-SbpSignature(edge's cost less than threshold).
  if (common_edge) {
    double min_cost = GetMaxVal<float>();
    for (const auto& row : common_edge->Cost) {
      for (const double& c : row) min_cost = std::min(min_cost, c);
    }
    // If there is no one case can choose, we will blow up
    for (int32_t i = 0; i < first->Cost.size(); i++) {
      for (int32_t j = 0; j < second->Cost.size(); j++) {
        const double edge_cost =
            common_edge->StartNode == first ? common_edge->Cost[i][j] : common_edge->Cost[j][i];
        if (edge_cost < cut_cost) {
          MergedSigId2ChildrenSigId.emplace_back(std::make_pair(i, j));
          Cost.emplace_back(edge_cost + first->Cost[i] + second->Cost[j]);
        }
      }
    }
    CHECK(MergedSigId2ChildrenSigId.size() > 0)
        << "0 size for merge child edge, min cost: " << min_cost;
  } else {
    for (int32_t i = 0; i < first->Cost.size(); i++) {
      for (int32_t j = 0; j < second->Cost.size(); j++) {
        MergedSigId2ChildrenSigId.emplace_back(std::make_pair(i, j));
        Cost.emplace_back(first->Cost[i] + second->Cost[j]);
      }
    }
  }

  // Initialize default sbp choice
  // If the original sbp pair does not go through, then use 0 as default.
  FinalSbpSignatureId = 0;
  // Track the original strategy
  for (int32_t sig_id = 0; sig_id < MergedSigId2ChildrenSigId.size(); sig_id++) {
    if (MergedSigId2ChildrenSigId[sig_id].first == first->FinalSbpSignatureId
        && MergedSigId2ChildrenSigId[sig_id].second == second->FinalSbpSignatureId) {
      FinalSbpSignatureId = sig_id;
    }
  }

  // Merge EdgesIn
  EdgesIn.reserve(first->EdgesIn.size() + second->EdgesIn.size());
  EdgesIn.insert(EdgesIn.end(), first->EdgesIn.begin(), first->EdgesIn.end());
  EdgesIn.insert(EdgesIn.end(), second->EdgesIn.begin(), second->EdgesIn.end());
  // Merge EdgesOut
  EdgesOut.reserve(first->EdgesOut.size() + second->EdgesOut.size());
  EdgesOut.insert(EdgesOut.end(), first->EdgesOut.begin(), first->EdgesOut.end());
  EdgesOut.insert(EdgesOut.end(), second->EdgesOut.begin(), second->EdgesOut.end());
  // Merge SbpEdge Cost
  for (SbpEdge<SbpSignature>*& this_edge : first->EdgesIn) {
    this_edge->DuplicateCost(false, true, MergedSigId2ChildrenSigId);
    this_edge->EndNode = this;
  }
  for (SbpEdge<SbpSignature>*& this_edge : first->EdgesOut) {
    this_edge->DuplicateCost(true, true, MergedSigId2ChildrenSigId);
    this_edge->StartNode = this;
  }
  for (SbpEdge<SbpSignature>*& this_edge : second->EdgesIn) {
    this_edge->DuplicateCost(false, false, MergedSigId2ChildrenSigId);
    this_edge->EndNode = this;
  }
  for (SbpEdge<SbpSignature>*& this_edge : second->EdgesOut) {
    this_edge->DuplicateCost(true, false, MergedSigId2ChildrenSigId);
    this_edge->StartNode = this;
  }
  // Remove edges from original nodes
  first->EdgesIn.clear();
  first->EdgesOut.clear();
  second->EdgesIn.clear();
  second->EdgesOut.clear();

  // Move edges between two nodes to each half node
  for (int32_t k = EdgesOut.size() - 1; k >= 0; k--) {
    if (EdgesOut[k]->EndNode == this) {
      // Remove this edge from EdgesOut and EdgesIn and put it inside the node
      CheckAndRemoveFrom<SbpEdge<SbpSignature>*>(EdgesIn, EdgesOut[k]);
      first->EdgesOut.emplace_back(EdgesOut[k]);
      second->EdgesIn.emplace_back(EdgesOut[k]);
      RemoveFrom<SbpEdge<SbpSignature>*>(EdgesOut, k);
    }
  }
}

template<class SbpSignature>
void SbpNode<SbpSignature>::FindLowOrderValue(
    const std::function<int32_t()>& CalcOrderValue4SbpSig) {
  LowOrderValue = 0;
  for (int32_t i = 0; i < OrderValue.size(); i++) {
    OrderValue[i] = CalcOrderValue4SbpSig();
    if (OrderValue[i] < LowOrderValue) { LowOrderValue = OrderValue[i]; }
  }
};

template<class SbpSignature>
void SbpNode<SbpSignature>::InitializeSbp(const std::function<int32_t()>& CalcOrderValue4SbpSig,
                                          std::vector<SbpSignature*> GlobalSbpSignatureList) {
  GlobalSbpSigSize = GlobalSbpSignatureList.size();
  OrderValue.resize(GlobalSbpSigSize);

  FindLowOrderValue(CalcOrderValue4SbpSig);

  SbpSignatureList.clear();
  for (int32_t sbp = 0; sbp < OrderValue.size(); sbp++) {
    if (OrderValue[sbp] == LowOrderValue) {
      SbpSignatureList.emplace_back(GlobalSbpSignatureList[sbp]);
    }
  }
  Cost.resize(SbpSignatureList.size());
};

template<class SbpSignature>
void SbpNode<SbpSignature>::InitializeSbp() {
  GlobalSbpSigSize = SbpSignatureObjList.size();
  SbpSignatureList.clear();
  for (int32_t i = 0; i < SbpSignatureObjList.size(); i++) {
    SbpSignatureList.emplace_back(&(SbpSignatureObjList[i]));
  }
  Cost.resize(SbpSignatureList.size());
};

template<class SbpSignature>
void SbpNode<SbpSignature>::ComputeCost(
    const std::function<double(SbpNode<SbpSignature>*, SbpSignature*)>& SbpComputationCost) {
  Cost.resize(SbpSignatureList.size());
  for (int32_t sbp = 0; sbp < SbpSignatureList.size(); sbp++) {
    Cost[sbp] = SbpComputationCost(this, SbpSignatureList[sbp]);
  }
};

// Let one node point to another
template<class SbpSignature>
void StartPointToEnd(SbpNode<SbpSignature>* start_node, SbpNode<SbpSignature>* end_node) {
#ifdef DEBUG_ALGORITHM_
  start_node->NodesOut.emplace_back(end_node);
  end_node->NodesIn.emplace_back(start_node);
#endif  // DEBUG_ALGORITHM_
  // generate the edge between them
  SbpEdge<SbpSignature>* e = new SbpEdge<SbpSignature>(start_node, end_node);
  start_node->EdgesOut.emplace_back(e);
  end_node->EdgesIn.emplace_back(e);
};

template<class SbpSignature>
void SbpNode<SbpSignature>::PointFrom(SbpNode<SbpSignature>* start_node) {
  StartPointToEnd(start_node, this);
};

template<class SbpSignature>
void SbpNode<SbpSignature>::PointTo(SbpNode<SbpSignature>* end_node) {
  StartPointToEnd(this, end_node);
};

template<class SbpSignature>
void SbpNode<SbpSignature>::SummarizeCost() {
  if (Children.size() == ChildNodeSbpSig.size()) { return; }
  int32_t PreviousChildrenSize = ChildNodeSbpSig.size();
  ChildNodeSbpSig.resize(Children.size());
  // Only deal with new Children
  for (int32_t child = PreviousChildrenSize; child < Children.size(); child++) {
    ChildNodeSbpSig[child].resize(Cost.size());

    for (int32_t sbp_this = 0; sbp_this < Cost.size(); sbp_this++) {
      double MinCost = 0, CurrCost;
      for (int32_t sbp_child = 0; sbp_child < Children[child]->Cost.size(); sbp_child++) {
        if (Children[child]->EdgesIn.size()) {
          // edge in graph: father -> child
          CurrCost = Children[child]->EdgesIn[0]->Cost[sbp_this][sbp_child]
                     + Children[child]->Cost[sbp_child];

        } else {
          // edge in graph: child -> father
          CurrCost = Children[child]->EdgesOut[0]->Cost[sbp_child][sbp_this]
                     + Children[child]->Cost[sbp_child];
        }
        // update MinCost with fixed SbpSignature for this node and child node
        if (sbp_child == 0 || CurrCost < MinCost) {
          MinCost = CurrCost;
          ChildNodeSbpSig[child][sbp_this] = sbp_child;
        }
      }
      // Add the cost for child node to this node
      Cost[sbp_this] += MinCost;
    }
  }
}

template<class SbpSignature>
void SbpNode<SbpSignature>::FinalizeSbp() {
  if (!HalfNode.empty()) {
    // Finalize Sbp of merged nodes
    HalfNode[0]->FinalSbpSignatureId = MergedSigId2ChildrenSigId[FinalSbpSignatureId].first;
    HalfNode[1]->FinalSbpSignatureId = MergedSigId2ChildrenSigId[FinalSbpSignatureId].second;
  }

  // Finalize Sbp of Children
  for (int32_t i = 0; i < Children.size(); i++) {
    Children[i]->FinalSbpSignatureId = ChildNodeSbpSig[i][this->FinalSbpSignatureId];
  }

  // Finalize Sbp of HalfNode Attachment
  if (!HalfNode.empty()) {
    HalfNode[0]->FinalizeSbp();
    HalfNode[1]->FinalizeSbp();
  }

  // Finalize Sbp of edges in EdgesOut
  for (const auto& edge_out : EdgesOut) edge_out->FinalizeSbp();

  // Finalize Sbp again in case of the node on the other side is not finalized
  // yet. This may happen when Two side of an edge merged into two larger nodes
  // and this edge is just a sub edge.
  for (const auto& edge_in : EdgesIn) edge_in->FinalizeSbp();

  // Finalize Sbp of Children Attachment
  for (int32_t i = 0; i < Children.size(); i++) {
    Children[i]->FinalizeSbp();
    for (const auto& edge_in : Children[i]->EdgesIn) edge_in->FinalizeSbp();
  }
}

template<class SbpSignature>
double SbpNode<SbpSignature>::GreedyStrategy() {
  // Current Cost, Minimum Cost, Cost with original sbp
  double CurrCost, MinCost, OrgCost;
  OrgCost = EvalNbhCost();
  MinCost = OrgCost;
  int32_t MinSbp = FinalSbpSignatureId;
  for (int32_t sbp = 0; sbp < Cost.size(); sbp++) {
    FinalSbpSignatureId = sbp;
    CurrCost = EvalNbhCost();
    if (CurrCost < MinCost) {
      MinCost = CurrCost;
      MinSbp = sbp;
    }
  }
  FinalSbpSignatureId = MinSbp;
  return MinCost - OrgCost;
}

template<class SbpSignature>
double SbpNode<SbpSignature>::EvalNbhCost() {
  // Current Cost, Minimum Cost, Cost with original sbp
  double CurrCost = Cost[FinalSbpSignatureId];
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    CurrCost += this_edge->Cost[this_edge->StartNode->FinalSbpSignatureId][FinalSbpSignatureId];
  }
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) {
    CurrCost += this_edge->Cost[FinalSbpSignatureId][this_edge->EndNode->FinalSbpSignatureId];
  }
  return CurrCost;
}

template<class SbpSignature>
double SbpNode<SbpSignature>::EvalOutNbhCost(
    std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id) {
  // check if this node is in the node list
  CHECK(NodeListId >= 0) << "Compute out cost for a node out of the node list" << std::endl;
  // Cost with original sbp
  double CurrCost = Cost[FinalSbpSignatureId];
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    // if the start node is not in the neighborhood
    if (NodeListId2nbh_id.find(this_edge->StartNode->NodeListId) == NodeListId2nbh_id.end()) {
      CurrCost += this_edge->Cost[this_edge->StartNode->FinalSbpSignatureId][FinalSbpSignatureId];
    }
  }
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) {
    // if the end node is not in the neighborhood
    if (NodeListId2nbh_id.find(this_edge->EndNode->NodeListId) == NodeListId2nbh_id.end()) {
      CurrCost += this_edge->Cost[FinalSbpSignatureId][this_edge->EndNode->FinalSbpSignatureId];
    }
  }
  return CurrCost;
}

// Compute the cost between this node and adjacent nodes with a lower order
template<class SbpSignature>
double SbpNode<SbpSignature>::EvalInNbhCost(std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id,
                                            std::vector<int32_t>& nbh_id2order) {
  // check if this node is in the node list
  CHECK(NodeListId >= 0) << "Compute in cost for a node out of the node list" << std::endl;
  // check if the node is in the neighborhood
  auto this_it = NodeListId2nbh_id.find(NodeListId);
  CHECK(this_it != NodeListId2nbh_id.end())
      << "Compute in cost for a node out of the neighborhood" << std::endl;
  // Compute the minimum cost between this node and adjacent nodes with a lower order
  int32_t order = nbh_id2order[this_it->second];
  double CurrCost = 0;
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    auto it = NodeListId2nbh_id.find(this_edge->StartNode->NodeListId);
    // if the start node is in the neighborhood
    if (it != NodeListId2nbh_id.end() && nbh_id2order[it->second] < order) {
      CurrCost += this_edge->Cost[this_edge->StartNode->FinalSbpSignatureId][FinalSbpSignatureId];
      // End this function and return infinity.
      if (CurrCost > cut_cost) { return GetMaxVal<float>(); }
    }
  }
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) {
    auto it = NodeListId2nbh_id.find(this_edge->EndNode->NodeListId);
    // if the end node is in the neighborhood
    if (it != NodeListId2nbh_id.end() && nbh_id2order[it->second] < order) {
      CurrCost += this_edge->Cost[FinalSbpSignatureId][this_edge->EndNode->FinalSbpSignatureId];
      if (CurrCost > cut_cost) { return GetMaxVal<float>(); }
    }
  }
  return CurrCost;
}

template<class SbpSignature>
double SbpNode<SbpSignature>::EvalMinInNbhCost(
    std::unordered_map<int32_t, int32_t>& NodeListId2nbh_id, std::vector<int32_t>& nbh_id2order) {
  // check if this node is in the node list
  CHECK(NodeListId >= 0) << "Compute out cost for a node out of the node list" << std::endl;
  // check if the node is in the neighborhood
  auto this_it = NodeListId2nbh_id.find(NodeListId);
  CHECK(this_it != NodeListId2nbh_id.end())
      << "Compute out cost for a node out of the neighborhood" << std::endl;
  // Compute the minimum cost between this node and adjacent nodes with a higher order
  int32_t order = nbh_id2order[this_it->second];
  double CurrCost = 0;
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    auto it = NodeListId2nbh_id.find(this_edge->StartNode->NodeListId);
    // if the start node is in the neighborhood
    if (it != NodeListId2nbh_id.end() && nbh_id2order[it->second] > order) {
      CurrCost += this_edge->GetMinCost();
    }
  }
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) {
    auto it = NodeListId2nbh_id.find(this_edge->EndNode->NodeListId);
    // if the end node is in the neighborhood
    if (it != NodeListId2nbh_id.end() && nbh_id2order[it->second] > order) {
      CurrCost += this_edge->GetMinCost();
    }
  }
  return CurrCost;
}

template<class SbpSignature>
void SbpNode<SbpSignature>::OneRingNeighborhood(std::vector<int32_t>& nbh_1ring) {
  nbh_1ring.resize(EdgesIn.size() + EdgesOut.size() + 1);
  int32_t nbh_id = 0;
  nbh_1ring[nbh_id] = NodeListId;
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    nbh_id++;
    nbh_1ring[nbh_id] = this_edge->StartNode->NodeListId;
  }
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) {
    nbh_id++;
    nbh_1ring[nbh_id] = this_edge->EndNode->NodeListId;
  }
}

// Get the n ring neighborhood of this node
// Pre-allocate buffer, which will be faster.
template<class SbpSignature>
void SbpNode<SbpSignature>::NRingNeighborhood(int32_t n, std::vector<int32_t>& nbh_nring,
                                              std::vector<int32_t>& nbh_1ring,
                                              std::vector<SbpNode<SbpSignature>*>& NodeList,
                                              std::vector<bool>& node_tags) {
  // Initialize 0 ring
  if (n <= 0) { n = 0; }
  nbh_nring.resize(1);
  nbh_nring[0] = NodeListId;
  node_tags[NodeListId] = true;
  int32_t l = 0, r;
  // do ring expansion for n times
  for (int32_t i = 0; i < n; i++) {
    for (r = nbh_nring.size(); l < r; l++) {
      NodeList[nbh_nring[l]]->OneRingNeighborhood(nbh_1ring);
      for (auto nbh_id : nbh_1ring) {
        if (!node_tags[nbh_id]) {
          nbh_nring.push_back(nbh_id);
          node_tags[nbh_id] = true;
        }
      }
    }
  }
  // Recover false for buffer
  for (auto nbh_id : nbh_nring) node_tags[nbh_id] = false;
}

// Detect and spread overlaps for EdgesIn.
template<class SbpSignature>
void SbpNode<SbpSignature>::DetectSpreadOverlap(double max_1_comp_cost, double max_2_comp_cost,
                                                int32_t max_1_id, double min_ratio) {
  // min_ratio should be less than 1.0
  CHECK(min_ratio < 1.0) << "Wrong overlap ratio here!" << std::endl;
  // If one layer have multiple nodes, the overlap occurs.
  // We need to skip sbp proxy and single node at each layer.
  // Actually, it is skipped before this function.

  // skip it if empty EdgesOut
  if (EdgesOut.empty()) { return; }

  // total maximum copy cost of outcoming edges
  double total_copy_cost = 0.0;
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) { total_copy_cost += this_edge->GetMaxCost(); }
  // maximum of the computation cost of other operators
  double max_comp_cost;
  if (id == max_1_id) {
    max_comp_cost = max_2_comp_cost;
  } else {
    max_comp_cost = max_1_comp_cost;
  }
  // Use the ratio between the total copy cost and maximum computation cost as ratio
  double overlap_ratio;
  if (max_comp_cost >= total_copy_cost) {
    overlap_ratio = min_ratio;
  } else {
    overlap_ratio = 1.0 - (1.0 - min_ratio) * max_comp_cost / total_copy_cost;
  }

  // Set up overlap ratio for the outcoming edges
  for (SbpEdge<SbpSignature>* this_edge : EdgesOut) {
    this_edge->DetectSpreadOverlap(overlap_ratio);
  }
}
// Detect and spread overlaps for sbp proxy.
template<class SbpSignature>
void SbpNode<SbpSignature>::DetectSpreadOverlap(double overlap_ratio) {
  if (op_node) { return; }
  if (overlap_ratio < 1.0) {
    // overlap ratio should be non-negative
    if (overlap_ratio < 0.0) { overlap_ratio = 0.0; }
    // double check for sbp proxy
    CHECK(EdgesIn.size() == 1) << "Multiple incoming edges for sbp proxy" << std::endl;
    EdgesIn[0]->DetectSpreadOverlap(overlap_ratio);
  }
}

// Get or compute the minimum layer of this node
template<class SbpSignature>
int32_t SbpNode<SbpSignature>::GetMinLayer(
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node) {
  if (MinLayer >= 0) { return MinLayer; }
  if (!op_node) { return MinLayer; }
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    int32_t producer_min_layer = this_edge->StartNode->GetMinLayer(op_name2sbp_node);
    if (producer_min_layer > MinLayer) { MinLayer = producer_min_layer; }
  }
  for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      int32_t producer_min_layer = it->second->GetMinLayer(op_name2sbp_node);
      if (producer_min_layer > MinLayer) { MinLayer = producer_min_layer; }
    }
  }
  return ++MinLayer;
}

// Spread the minimum layer to compute the maximum layer of producers
template<class SbpSignature>
void SbpNode<SbpSignature>::SpreadMaxLayer(
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node) {
  if (MinLayer <= 0) { return; }
  int32_t producer_max_lay = MinLayer - 1;
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    this_edge->StartNode->DropMaxLayer(producer_max_lay);
  }
  for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) { it->second->DropMaxLayer(producer_max_lay); }
  }
}

// Drop down the maximum layer with the minimum layer form consumer
template<class SbpSignature>
void SbpNode<SbpSignature>::DropMaxLayer(int32_t upper_bound) {
  if (upper_bound < MaxLayer || MaxLayer < 0) { MaxLayer = upper_bound; }
}
// Set MaxLayer = MinLayer if this node does not have any consumer
// This is the end of the whole graph
// We could also set it to be the maximum of the MinLayer in the graph. (It should be the same.)
template<class SbpSignature>
void SbpNode<SbpSignature>::LiftMaxLayer() {
  if (MaxLayer < MinLayer) { MaxLayer = MinLayer; }
}
// Set MaxLayer = upper_bound if this node does not have any consumer
template<class SbpSignature>
void SbpNode<SbpSignature>::LiftMaxLayer(int32_t upper_bound) {
  if (MaxLayer < MinLayer) { MaxLayer = upper_bound; }
}

// Get the minimum element in Cost
template<class SbpSignature>
double SbpNode<SbpSignature>::GetMinCost() {
  // Check the size of Cost
  CHECK(Cost.size() > 0) << "Cost not initialized!" << std::endl;
  // Compute the min_comp_cost
  return *std::min_element(Cost.begin(), Cost.end());
}

// Set the cut ratio
template<class SbpSignature>
double SbpNode<SbpSignature>::GetCutRatio() {
  double curr_cut_ratio = 1.0;
  for (auto* this_edge : EdgesIn) { curr_cut_ratio *= this_edge->GetCutRatio(); }
  for (auto* this_edge : EdgesOut) { curr_cut_ratio *= this_edge->GetCutRatio(); }
  return curr_cut_ratio;
}

// Judge if this node is on the mainstem
// If so, judge it for its producer/upstream nodes
template<class SbpSignature>
void SbpNode<SbpSignature>::SpreadMainstem(
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node) {
  // Skip it if this node is already judged.
  if (IfMainstem) { return; }
  // Skip sbp proxy. This is before we have proxy.
  if (MinLayer < 0) { return; }
  IfMainstem = true;
  // If I am in the mainstem, then all the children with (MinLayer >= my layer id - 1) would be
  // considered as in the mainstem
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    if (this_edge->StartNode->MinLayer >= MinLayer - 1) {
      this_edge->StartNode->SpreadMainstem(op_name2sbp_node);
    }
  }
  for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end() && it->second->MinLayer >= MinLayer - 1) {
      it->second->SpreadMainstem(op_name2sbp_node);
    }
  }
}

// Count consumers and any downstream nodes defined by control edges
template<class SbpSignature>
void SbpNode<SbpSignature>::RaiseConsumerNum(
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node) {
  // Should clear it before running.
  // skip the proxy nodes and the sources
  if (MinLayer <= 0) { return; }
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) { this_edge->StartNode->counter++; }
  for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) { it->second->counter++; }
  }
}

// Compute the minimal available wait time for producers or upstream nodes
template<class SbpSignature>
void SbpNode<SbpSignature>::SpreadAvailWaitTime(
    std::vector<double>& mainstem_cost, std::vector<double>& acc_mainstem_cost,
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node, double wait_time,
    double transfer_cost) {
  // skip the proxy nodes and the sources
  if (MinLayer <= 0) { return; }
  // Have not finished spreading for consumers or downstream nodes or already visited.
  if (counter) { return; }
  if (IfMainstem) {
    // Nodes on the mianstem does not have any accumulate cost
    AccMainstemCost = 0;
  } else {
    if (AccMainstemCost < 0) {
      // Do not have any consumer or downstream node
      AccMainstemCost = acc_mainstem_cost[MinLayer - 1];
    } else {
      // Add the mainstem cost at this layer
      AccMainstemCost += mainstem_cost[MinLayer];
    }
  }

  // Reduce the wait time for EdgesIn, put the rest of the mainstem cost in the producers
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    CHECK(this_edge->WaitTime < 0)
        << "Double assgin values into WaitTime of this edge!" << std::endl;
    SbpNode<SbpSignature>* producer = this_edge->StartNode;
    // Accumulate the cost from the start node to this node
    double curr_mainstem_cost =
        AccMainstemCost + acc_mainstem_cost[producer->MinLayer] - acc_mainstem_cost[MinLayer - 1];
    if (curr_mainstem_cost >= wait_time) {
      // Remain cost in the mainstem is able to cover all the wait time
      this_edge->WaitTime = 0.0;
      curr_mainstem_cost -= wait_time;
    } else {
      // Remain cost in the mainstem can only cover partial wait time
      this_edge->WaitTime = wait_time - curr_mainstem_cost;
      curr_mainstem_cost = 0.0;
    }
    // Reducing non-matching edges
    // For example:
    // (1) P->S0->S0->S0->B
    // (2) p->B->B->B->B
    // We would use (2) when the tensor is relatively tiny.
    this_edge->WaitTime += transfer_cost;
    // Do not inherit mainstem cost for nodes on the mainstem
    if (!producer->IfMainstem) {
      // Inherit the minimal of the mainstem cost from consumers
      producer->DropAvailWaitTime(curr_mainstem_cost);
    }
    producer->counter--;
    producer->SpreadAvailWaitTime(mainstem_cost, acc_mainstem_cost, op_name2sbp_node, wait_time,
                                  transfer_cost);
  }
  // Put the rest the mainstem cost in the upstream nodes.
  for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      SbpNode<SbpSignature>* producer = it->second;
      // Do not inherit mainstem cost for nodes on the mainstem
      if (!producer->IfMainstem) {
        // Accumulate the cost from the start node to this node
        double curr_mainstem_cost = AccMainstemCost + acc_mainstem_cost[producer->MinLayer]
                                    - acc_mainstem_cost[MinLayer - 1];
        // Inherit the minimal of the mainstem cost from consumers
        producer->DropAvailWaitTime(curr_mainstem_cost);
      }
      producer->counter--;
      producer->SpreadAvailWaitTime(mainstem_cost, acc_mainstem_cost, op_name2sbp_node, wait_time,
                                    transfer_cost);
    }
  }
  // Set counter to be -1, do not visit it again.
  counter--;
}

// Drop down the available wait time with the minimum cost from downstreams
template<class SbpSignature>
void SbpNode<SbpSignature>::DropAvailWaitTime(double curr_mainstem_cost) {
  if (AccMainstemCost < 0.0 || AccMainstemCost > curr_mainstem_cost) {
    AccMainstemCost = curr_mainstem_cost;
  }
}

// Assemble copy cost for all the incoming edges
template<class SbpSignature>
void SbpNode<SbpSignature>::InitializeCopyCost(bool compute_cost, bool use_sbp_collector_) {
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    const auto* sbp_node_producer = this_edge->StartNode;
    oneflow::OpNode* producer = sbp_node_producer->op_node;

    // skip it if proxy
    if (use_sbp_collector_ && !producer) { continue; }

    // look through input blobs
    for (const std::string& ibn : op_node->op().input_bns()) {
      if (producer->op().op_name() == op_node->SrcNode4Ibn(ibn).op().op_name()) {
        this_edge->InitializeCopyCost(ibn, compute_cost, use_sbp_collector_);
      }
    }
  }
}

// Reduce and set the wait time for op in the mainstem
template<class SbpSignature>
void SbpNode<SbpSignature>::SetMainstemWaitTime(double mainstem_wait_time) {
  // only reduce the wait time for operators in the mainstem
  if (IfMainstem) {
    // Reduce the wait time for EdgesOut
    for (SbpEdge<SbpSignature>* edge_out : EdgesOut) {
      if (edge_out->WaitTime < 0.0 || edge_out->WaitTime > mainstem_wait_time) {
        edge_out->WaitTime = mainstem_wait_time;
      }
    }
    // Might reduce it for EdgesIn
  }
}

// Drop down the maximum layer with the minimum layer form consumer
template<class SbpSignature>
void SbpNode<SbpSignature>::DropTributaryLayer(int32_t upper_bound) {
  if (upper_bound < TributaryLayer || TributaryLayer < 0) { TributaryLayer = upper_bound; }
}

// Compute maximum layer for tributaries
template<class SbpSignature>
void SbpNode<SbpSignature>::SpreadTributaryLayer(
    oneflow::HashMap<std::string, SbpNode<SbpSignature>*>& op_name2sbp_node) {
  if (counter || MinLayer <= 0) { return; }
  int32_t producer_max_lay;
  if (IfMainstem) {
    producer_max_lay = MinLayer - 1;
  } else {
    // On a tributary, the operator could be run later.
    producer_max_lay = TributaryLayer;
    // producer_max_lay = TributaryLayer - 1;
  }
  for (SbpEdge<SbpSignature>* this_edge : EdgesIn) {
    this_edge->StartNode->DropTributaryLayer(producer_max_lay);
    if (--this_edge->StartNode->counter == 0) {
      this_edge->StartNode->SpreadTributaryLayer(op_name2sbp_node);
    }
  }
  for (const auto& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
    auto it = op_name2sbp_node.find(ctrl_in_op_name);
    if (it != op_name2sbp_node.end()) {
      it->second->DropTributaryLayer(producer_max_lay);
      if (--it->second->counter == 0) { it->second->SpreadTributaryLayer(op_name2sbp_node); }
    }
  }
  counter--;
}

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // SBP_NODE_H_
