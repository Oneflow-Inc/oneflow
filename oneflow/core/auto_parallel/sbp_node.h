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

namespace Algorithm {

template<class SbpSignature>
class SbpEdge;

template<class SbpSignature>
class SbpNode {
 public:
  // Data Structure

  // compound edge in
  std::vector<SbpEdge<SbpSignature> *> EdgesIn;
  // compound edge out
  std::vector<SbpEdge<SbpSignature> *> EdgesOut;
  // Identity, use it to distinguish itself from node set
  int32_t id;
  // Matrix Dimension
  // Y = XW, X is MatDim[0]-by-MatDim[1], W is MatDim[1]-by-MatDim[2]
  int32_t MatDim[3];

  // We should use Sbp-signature for edge with lowest OrderValue
  std::vector<int32_t> OrderValue;
  // Lowest OrderValue
  int32_t LowOrderValue;
  // Available SbpSignature pointer for this node
  std::vector<SbpSignature *> SbpSignatureList;
  // Available SbpSignature object for this node
  std::vector<SbpSignature> SbpSignatureObjList;
  // Global SbpSignature List Size
  int32_t GlobalSbpSigSize;
  // Decide to use SbpSignature with this id
  int32_t FinalSbpSignatureId;
  // Location in NodeList
  int32_t NodeListId = -1;

  // Child node list
  std::vector<SbpNode<SbpSignature> *> Children;
  // SbpSignature for each child node when using specific SbpSignature for this
  // node Its dimension is Number of Child Nodes * Number of Available
  // SbpSignatures for this node
  std::vector<std::vector<int32_t>> ChildNodeSbpSig;

  // Merge two nodes into this compound node
  std::vector<SbpNode<SbpSignature> *> HalfNode;
  // We should delete those merged-signatures which has very large cost for speed up
  // New SbpSignatureList index map to each HalfNode's sig_index
  std::vector<std::pair<int32_t, int32_t>> MergedSigId2ChildrenSigId;

  // Cost[sbp] is Computation Cost when using SbpSignatureList[sbp]
  std::vector<double> Cost;

#ifdef DEBUG_ALGORITHM_

  // original edge out
  std::vector<SbpNode *> NodesOut;
  // original cost for edge out
  std::vector<std::vector<std::vector<double>>> OriginCostOut;
  // original edge in
  std::vector<SbpNode *> NodesIn;
  // Original Cost
  std::vector<double> OriginCost;

  // Current Degree is a tag used for Topological ordering
  int32_t CurrDeg;
#endif  // DEBUG_ALGORITHM_

  // functions
  SbpNode() { FinalSbpSignatureId = 0; }

  SbpNode(int32_t DataSize, int32_t InputParameterDim, int32_t OutputParameterDim) {
    MatDim[0] = DataSize;
    MatDim[1] = InputParameterDim;
    MatDim[2] = OutputParameterDim;
  }

  // This constructor is to merge two node into one
  SbpNode(SbpNode<SbpSignature> *first, SbpNode<SbpSignature> *second);

  ~SbpNode() {
    for (auto &edge_out : EdgesOut) { delete edge_out; }
    for (auto &childnode : Children) {
      if (childnode->EdgesIn.size()) { delete childnode->EdgesIn[0]; }
      delete childnode;
    }
    for (auto &half_node : HalfNode) { delete half_node; }
  }

  // another node point to this node
  void PointFrom(SbpNode<SbpSignature> *start_node);
  // this node point to another node
  void PointTo(SbpNode<SbpSignature> *end_node);

  // initialize the OrderValue and Find the lowest one
  void FindLowOrderValue(const std::function<int32_t()> &CalcOrderValue4SbpSig);
  // Initialize SbpSignature
  void InitializeSbp(const std::function<int32_t()> &CalcOrderValue4SbpSig,
                     std::vector<SbpSignature *> GlobalSbpSignatureList);
  // Initialize SbpSignature from Signature Objects
  void InitializeSbp();
  // Compute Computation Cost
  void ComputeCost(
      const std::function<double(SbpNode<SbpSignature> *, SbpSignature *)> &SbpComputationCost);
  // Decide to use this SbpSignature
  SbpSignature *FinalSbpSignature() {
    if (SbpSignatureList.empty()) return NULL;
    return SbpSignatureList[FinalSbpSignatureId];
  };

  // Recompute Computation Cost after adding child nodes in it
  void SummerizeCost();
  // Determine Final SbpSignature for attachment of this node
  void FinalizeSbp();
  // Use Greedy Strategy to pick the sbp signature with minimum cost for this
  // node You should have an initial strategy before running this
  double GreedyStrategy();
  // Evaluate summery of cost in 1-ring neighborhood.
  double EvalNbhCost();

};  // class SbpNode
}  // namespace Algorithm

// function in cpp. Should be put in one file due to use of template
// Otherwise we will need to declare specific template at the end of cpp file.
namespace Algorithm {

// this function is to remove the i-th element from a vector in Constant time.
// the vector should not care about ordering.
// Be more careful about this function. Make sure that the traveling order of
// the vector goes from back to front.
template<class T>
void RemoveFrom(std::vector<T> &v, int32_t i) {
  v[i] = v.back();
  v.pop_back();
}

template<class T>
void CheckAndRemoveFrom(std::vector<T> &v, T &t) {
  for (int32_t i = v.size() - 1; i >= 0; i--) {
    if (v[i] == t) {
      RemoveFrom<T>(v, i);
      break;
    }
  }
}

template<class SbpSignature>
SbpNode<SbpSignature>::SbpNode(SbpNode<SbpSignature> *first, SbpNode<SbpSignature> *second) {
  HalfNode.resize(2);
  HalfNode[0] = first;
  HalfNode[1] = second;

  // Get the edge between first and second
  // NOTE: It must zero or one edge between them
  SbpEdge<SbpSignature> *common_edge = nullptr;
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
    double edge_threshold = 3e38;
    double min_cost = 1e100;
    for (const auto &row : common_edge->Cost) {
      for (const double &c : row) std::min(min_cost, c);
    }
    // If there is no one case can choose, we will choose pairs whose cost in [min_cost,
    // min_cost*10]
    edge_threshold = min_cost > edge_threshold ? min_cost * 10 : edge_threshold;
    for (int32_t i = 0; i < first->Cost.size(); i++) {
      for (int32_t j = 0; j < second->Cost.size(); j++) {
        const double edge_cost =
            common_edge->StartNode == first ? common_edge->Cost[i][j] : common_edge->Cost[j][i];
        if (edge_cost < edge_threshold) {
          MergedSigId2ChildrenSigId.emplace_back(std::make_pair(i, j));
          Cost.emplace_back(edge_cost + first->Cost[i] + second->Cost[j]);
        }
      }
    }
  } else {
    for (int32_t i = 0; i < first->Cost.size(); i++) {
      for (int32_t j = 0; j < second->Cost.size(); j++) {
        MergedSigId2ChildrenSigId.emplace_back(std::make_pair(i, j));
        Cost.emplace_back(first->Cost[i] + second->Cost[j]);
      }
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
  for (SbpEdge<SbpSignature> *&this_edge : first->EdgesIn) {
    this_edge->DuplicateCost(false, true, MergedSigId2ChildrenSigId);
    this_edge->EndNode = this;
  }
  for (SbpEdge<SbpSignature> *&this_edge : first->EdgesOut) {
    this_edge->DuplicateCost(true, true, MergedSigId2ChildrenSigId);
    this_edge->StartNode = this;
  }
  for (SbpEdge<SbpSignature> *&this_edge : second->EdgesIn) {
    this_edge->DuplicateCost(false, false, MergedSigId2ChildrenSigId);
    this_edge->EndNode = this;
  }
  for (SbpEdge<SbpSignature> *&this_edge : second->EdgesOut) {
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
      CheckAndRemoveFrom<SbpEdge<SbpSignature> *>(EdgesIn, EdgesOut[k]);
      first->EdgesOut.emplace_back(EdgesOut[k]);
      second->EdgesIn.emplace_back(EdgesOut[k]);
      RemoveFrom<SbpEdge<SbpSignature> *>(EdgesOut, k);
    }
  }

  // Initialize default sbp choice
  FinalSbpSignatureId = 0;
}

template<class SbpSignature>
void SbpNode<SbpSignature>::FindLowOrderValue(
    const std::function<int32_t()> &CalcOrderValue4SbpSig) {
  LowOrderValue = 0;
  for (int32_t i = 0; i < OrderValue.size(); i++) {
    OrderValue[i] = CalcOrderValue4SbpSig();
    if (OrderValue[i] < LowOrderValue) LowOrderValue = OrderValue[i];
  }
};

template<class SbpSignature>
void SbpNode<SbpSignature>::InitializeSbp(const std::function<int32_t()> &CalcOrderValue4SbpSig,
                                          std::vector<SbpSignature *> GlobalSbpSignatureList) {
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
  for (int32_t sbp = 0; sbp < SbpSignatureObjList.size(); sbp++) {
    SbpSignatureList.emplace_back(&(SbpSignatureObjList[sbp]));
  }
  Cost.resize(SbpSignatureList.size());
};

template<class SbpSignature>
void SbpNode<SbpSignature>::ComputeCost(
    const std::function<double(SbpNode<SbpSignature> *, SbpSignature *)> &SbpComputationCost) {
  Cost.resize(SbpSignatureList.size());
  for (int32_t sbp = 0; sbp < SbpSignatureList.size(); sbp++) {
    Cost[sbp] = SbpComputationCost(this, SbpSignatureList[sbp]);
  }
};

// Let one node point to another
template<class SbpSignature>
void StartPointToEnd(SbpNode<SbpSignature> *start_node, SbpNode<SbpSignature> *end_node) {
#ifdef DEBUG_ALGORITHM_
  start_node->NodesOut.emplace_back(end_node);
  end_node->NodesIn.emplace_back(start_node);
#endif  // DEBUG_ALGORITHM_
  // generate the edge between them
  SbpEdge<SbpSignature> *e = new SbpEdge<SbpSignature>(start_node, end_node);
  start_node->EdgesOut.emplace_back(e);
  end_node->EdgesIn.emplace_back(e);
};

template<class SbpSignature>
void SbpNode<SbpSignature>::PointFrom(SbpNode<SbpSignature> *start_node) {
  StartPointToEnd(start_node, this);
};

template<class SbpSignature>
void SbpNode<SbpSignature>::PointTo(SbpNode<SbpSignature> *end_node) {
  StartPointToEnd(this, end_node);
};

template<class SbpSignature>
void SbpNode<SbpSignature>::SummerizeCost() {
  if (Children.size() == ChildNodeSbpSig.size()) return;
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
  for (const auto &edge_out : EdgesOut) edge_out->FinalizeSbp();

  // Finalize Sbp again in case of the node on the other side is not finalized
  // yet. This may happen when Two side of an edge merged into two larger nodes
  // and this edge is just a sub edge.
  for (const auto &edge_in : EdgesIn) edge_in->FinalizeSbp();

  // Finalize Sbp of Children Attachment
  for (int32_t i = 0; i < Children.size(); i++) {
    Children[i]->FinalizeSbp();
    for (const auto &edge_in : Children[i]->EdgesIn) edge_in->FinalizeSbp();
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
  for (SbpEdge<SbpSignature> *this_edge : EdgesIn) {
    CurrCost += this_edge->Cost[this_edge->StartNode->FinalSbpSignatureId][FinalSbpSignatureId];
  }
  for (SbpEdge<SbpSignature> *this_edge : EdgesOut) {
    CurrCost += this_edge->Cost[FinalSbpSignatureId][this_edge->EndNode->FinalSbpSignatureId];
  }
  return CurrCost;
}

}  // namespace Algorithm

#endif  // SBP_NODE_H_