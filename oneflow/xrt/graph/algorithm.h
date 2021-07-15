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
#ifndef ONEFLOW_XRT_GRAPH_ALGORITHM_H_
#define ONEFLOW_XRT_GRAPH_ALGORITHM_H_

#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace algorithm {

template<typename GraphType>
struct GraphTypeTrait {
  typedef typename GraphType::NodeType* pNodeType;
  typedef typename GraphType::EdgeType* pEdgeType;
};

template<typename NodeType>
struct NodeTypeTrait {
  typedef typename NodeType::EdgeType* pEdgeType;
};

template<typename GraphType, typename UserFunc>
inline void TopologyVisit(GraphType& graph, UserFunc func) {
  typedef typename GraphTypeTrait<GraphType>::pNodeType pNodeType;
  typedef typename GraphTypeTrait<GraphType>::pEdgeType pEdgeType;

  util::Set<pNodeType> visited;
  util::Queue<pNodeType> visit_queue;
  for (pNodeType node : graph.Nodes()) {
    if (node->IsSourceNode()) {
      visit_queue.push(node);
      visited.insert(node);
    }
  }

  auto IsAllInputsVisited = [&](pNodeType node) -> bool {
    for (pEdgeType edge : node->in_edges()) {
      pNodeType start = edge->start();
      if (visited.count(start) == 0) { return false; }
    }
    return true;
  };

  while (!visit_queue.empty()) {
    pNodeType node = visit_queue.front();
    visit_queue.pop();
    {  // Run user function
      func(node);
    }
    for (pEdgeType edge : node->out_edges()) {
      pNodeType end = edge->end();
      if (IsAllInputsVisited(end) && visited.insert(end).second) { visit_queue.push(end); }
    }
  }
};

template<typename NodeType>
inline bool IsReachable(NodeType* start, NodeType* dest) {
  typedef NodeType* pNodeType;
  typedef typename NodeTypeTrait<NodeType>::pEdgeType pEdgeType;

  util::Set<pNodeType> visited_nodes;
  util::Stack<pNodeType> stack;
  for (pEdgeType edge : start->out_edges()) { stack.push(edge->end()); }

  while (!stack.empty()) {
    pNodeType node = stack.top();
    stack.pop();
    if (node == dest) { return true; }
    for (pEdgeType edge : node->out_edges()) {
      pNodeType end = edge->end();
      if (visited_nodes.insert(end).second) { stack.push(end); }
    }
  }
  return false;
}

}  // namespace algorithm
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_ALGORITHM_H_
