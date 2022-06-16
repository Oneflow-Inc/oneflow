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
#ifndef ONEFLOW_CORE_GRAPH_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_GRAPH_H_

#include <stack>
#include <bitset>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/graph/node.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

template<typename NodeType, typename EdgeType>
class Graph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Graph);
  Graph() = default;
  virtual ~Graph() = default;

  // For Each
  void ForEachNode(std::function<void(NodeType*)> NodeHandler) const;
  Maybe<void> MaybeForEachNode(std::function<Maybe<void>(NodeType*)> NodeHandler) const;
  // In case you want to change the topological structure during the node handler.
  // For example, adding/deleting a node or an edge.
  // Still, it might have bugs even if you use TopoForEachNodeDynamic.
  void TopoForEachNodeDynamic(std::function<void(NodeType*)> NodeHandler) const;
  void TopoForEachNode(std::function<void(NodeType*)> NodeHandler) const;
  Maybe<void> TopoForEachNodeDynamicWithErrorCaptured(
      std::function<Maybe<void>(NodeType*)> NodeHandler) const;
  Maybe<void> TopoForEachNodeWithErrorCaptured(
      std::function<Maybe<void>(NodeType*)> NodeHandler) const;
  void ReverseTopoForEachNode(std::function<void(NodeType*)> NodeHandler) const;
  void ForEachEdge(std::function<void(EdgeType*)> EdgeHandler) const;

  void SortedTopoForEachNode(std::function<bool(const EdgeType* lhs, const EdgeType* rhs)> LessThan,
                             std::function<void(NodeType*)> NodeHandler) const;

  void BfsForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
      const std::function<void(NodeType*)>& Handler) const;

  void DfsForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
      const std::function<void(NodeType*)>& Handler) const;

  void TopoForEachNodeDynamic(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  void TopoForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  void TopoForEachNode(
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  Maybe<void> TopoForEachNodeDynamicWithErrorCaptured(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<Maybe<void>(NodeType*)>& Handler) const;

  Maybe<void> TopoForEachNodeWithErrorCaptured(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<Maybe<void>(NodeType*)>& Handler) const;

  Maybe<void> TopoForEachNodeWithErrorCaptured(
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<Maybe<void>(NodeType*)>& Handler) const;

  void DfsTopoForEachNode(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  void DfsTopoForEachNodeSortByDistanceToSink(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
      const std::function<void(NodeType*)>& Handler) const;

  std::function<bool(const NodeType* src, const NodeType* dst)> MakePredicatorIsReachable() const;

  std::function<bool(const NodeType* src, const NodeType* dst)> MakePredicatorIsReachable(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
      const;

  void ForEachConnectedComponent(
      const std::function<void(const HashSet<NodeType*>&)>& Handler) const;

  void ForEachConnectedComponent(
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
      const std::function<void(const HashSet<NodeType*>&)>& Handler) const;

  void ForEachConnectedComponent(
      const std::function<void(const std::function<void(NodeType*)>&)>& ForEachNodeAsStart,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
      const std::function<void(const HashSet<NodeType*>&)>& Handler) const;

  // find first nontrivial strongly connected component
  std::unique_ptr<HashSet<NodeType*>> FindFirstNontrivialSCC(
      const std::list<NodeType*>& starts,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
      const;

  std::unique_ptr<HashSet<NodeType*>> FindFirstNontrivialSCC(
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
      const;

  std::unique_ptr<HashSet<NodeType*>> FindFirstNontrivialSCC() const;

  // Getters
  std::list<NodeType*> source_nodes() const;
  std::list<NodeType*> sink_nodes() const;
  NodeType* SoleSourceNode() const;
  NodeType* SoleSinkNode() const;
  NodeType* SoleNode() const;
  size_t node_num() const { return nodes_.size(); }
  size_t edge_num() const { return edges_.size(); }
  virtual const char* TypeName() const { return ""; }

  // Setters
  template<typename DerivedNodeType = NodeType>
  DerivedNodeType* NewNode();
  template<class... Args>
  EdgeType* NewEdge(Args&&... args);
  void AddAllocatedNode(NodeType*);
  void AddAllocatedEdge(EdgeType*);
  void DeleteNode(NodeType*);

  // ToDot
  template<typename StreamT>
  void ToDotWithStream(StreamT& out_stream) const;
  template<typename StreamT>
  void ToDotWithStream(const std::function<bool(NodeType*)>& IsNodeAllowed,
                       const std::function<bool(EdgeType*)>& IsEdgeAllowed,
                       const std::function<std::string(NodeType*)>& AddNodeAttribute,
                       const std::function<std::string(EdgeType*)>& AddEdgeAttribute,
                       StreamT& out_stream) const;
  void ToDotWithFilePath(const std::string& file_path) const;
  void ToDotWithFilePath(const std::function<std::string(NodeType*)>& AddNodeAttribute,
                         const std::function<std::string(EdgeType*)>& AddEdgeAttribute,
                         const std::string& file_path) const;
  void ToDotWithFilePath(const std::function<bool(NodeType*)>& IsNodeAllowed,
                         const std::function<bool(EdgeType*)>& IsEdgeAllowed,
                         const std::string& file_path) const;
  void ToDotWithAutoFilePath() const;

 private:
  std::unique_ptr<HashSet<NodeType*>> FindFirstNontrivialSCC(
      const std::function<void(const std::function<void(NodeType*)>&)>& ForEachStart,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
      const;

  // finish time first search
  void FfsForEachNode(
      const std::function<void(const std::function<void(NodeType*)>&)>& ForEachStart,
      const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
      const std::function<void(NodeType*)>& Handler) const;

  void FfsForEachNode(const std::function<void(NodeType*)>& Handler) const;

  std::vector<std::unique_ptr<NodeType>> nodes_;
  std::vector<std::unique_ptr<EdgeType>> edges_;
};

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachNode(std::function<void(NodeType*)> NodeHandler) const {
  for (auto& x : nodes_) { NodeHandler(x.get()); }
}

template<typename NodeType, typename EdgeType>
Maybe<void> Graph<NodeType, EdgeType>::MaybeForEachNode(
    std::function<Maybe<void>(NodeType*)> NodeHandler) const {
  for (auto& x : nodes_) { JUST(NodeHandler(x.get())); }
  return Maybe<void>::Ok();
}

template<typename NodeType, typename EdgeType>
std::list<NodeType*> Graph<NodeType, EdgeType>::source_nodes() const {
  std::list<NodeType*> ret;
  ForEachNode([&](NodeType* node) {
    if (node->in_edges().empty()) { ret.emplace_back(node); }
  });
  return ret;
}

template<typename NodeType, typename EdgeType>
std::list<NodeType*> Graph<NodeType, EdgeType>::sink_nodes() const {
  std::list<NodeType*> ret;
  ForEachNode([&](NodeType* node) {
    if (node->out_edges().empty()) { ret.emplace_back(node); }
  });
  return ret;
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleSourceNode() const {
  std::list<NodeType*> source_nodes_list = source_nodes();
  CHECK_EQ(source_nodes_list.size(), 1);
  return source_nodes_list.front();
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleSinkNode() const {
  std::list<NodeType*> sink_nodes_list = sink_nodes();
  CHECK_EQ(sink_nodes_list.size(), 1);
  return sink_nodes_list.front();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNodeDynamic(
    std::function<void(NodeType*)> NodeHandler) const {
  TopoForEachNodeDynamic(source_nodes(), &NodeType::ForEachNodeOnInEdge,
                         &NodeType::ForEachNodeOnOutEdge, NodeHandler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNode(std::function<void(NodeType*)> NodeHandler) const {
  CHECK_JUST(TopoForEachNodeWithErrorCaptured(&NodeType::ForEachNodeOnInEdge,
                                              &NodeType::ForEachNodeOnOutEdge, [&](NodeType* node) {
                                                NodeHandler(node);
                                                return Maybe<void>::Ok();
                                              }));
}

template<typename NodeType, typename EdgeType>
Maybe<void> Graph<NodeType, EdgeType>::TopoForEachNodeDynamicWithErrorCaptured(
    std::function<Maybe<void>(NodeType*)> NodeHandler) const {
  return TopoForEachNodeDynamicWithErrorCaptured(source_nodes(), &NodeType::ForEachNodeOnInEdge,
                                                 &NodeType::ForEachNodeOnOutEdge, NodeHandler);
}

template<typename NodeType, typename EdgeType>
Maybe<void> Graph<NodeType, EdgeType>::TopoForEachNodeWithErrorCaptured(
    std::function<Maybe<void>(NodeType*)> NodeHandler) const {
  return TopoForEachNodeWithErrorCaptured(&NodeType::ForEachNodeOnInEdge,
                                          &NodeType::ForEachNodeOnOutEdge, NodeHandler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::SortedTopoForEachNode(
    std::function<bool(const EdgeType* lhs, const EdgeType* rhs)> LessThan,
    std::function<void(NodeType*)> NodeHandler) const {
  ForEachNode([&](NodeType* node) { node->SortInOutEdges(LessThan); });
  TopoForEachNode(&NodeType::ForEachNodeOnSortedInEdge, &NodeType::ForEachNodeOnSortedOutEdge,
                  NodeHandler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ReverseTopoForEachNode(
    std::function<void(NodeType*)> NodeHandler) const {
  TopoForEachNode(&NodeType::ForEachNodeOnOutEdge, &NodeType::ForEachNodeOnInEdge, NodeHandler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachEdge(std::function<void(EdgeType*)> EdgeHandler) const {
  for (auto& x : edges_) {
    if (x->src_node() == nullptr && x->dst_node() == nullptr) { continue; }
    EdgeHandler(x.get());
  }
}

template<typename NodeType, typename EdgeType>
NodeType* Graph<NodeType, EdgeType>::SoleNode() const {
  CHECK_EQ(nodes_.size(), 1);
  return nodes_.front().get();
}

template<typename NodeType, typename EdgeType>
template<typename DerivedNodeType>
DerivedNodeType* Graph<NodeType, EdgeType>::NewNode() {
  DerivedNodeType* ret = new DerivedNodeType;
  AddAllocatedNode(ret);
  return ret;
}

template<typename NodeType, typename EdgeType>
template<class... Args>
EdgeType* Graph<NodeType, EdgeType>::NewEdge(Args&&... args) {
  EdgeType* ret = new EdgeType(std::forward<Args>(args)...);
  AddAllocatedEdge(ret);
  return ret;
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::AddAllocatedNode(NodeType* node) {
  nodes_.emplace_back(node);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::AddAllocatedEdge(EdgeType* edge) {
  edges_.emplace_back(edge);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DeleteNode(NodeType* node) {
  Erase<std::vector<std::unique_ptr<NodeType>>>(
      nodes_, [node](const std::unique_ptr<NodeType>& node_ptr) { return node_ptr.get() == node; });
}

template<typename NodeType, typename EdgeType>
template<typename StreamT>
void Graph<NodeType, EdgeType>::ToDotWithStream(StreamT& out_stream) const {
  ToDotWithStream([](NodeType*) { return true; }, [](EdgeType*) { return true; },
                  [](NodeType*) { return ""; }, [](EdgeType*) { return ""; }, out_stream);
}

template<typename NodeType, typename EdgeType>
template<typename StreamT>
void Graph<NodeType, EdgeType>::ToDotWithStream(
    const std::function<bool(NodeType*)>& IsNodeAllowed,
    const std::function<bool(EdgeType*)>& IsEdgeAllowed,
    const std::function<std::string(NodeType*)>& AddNodeAttribute,
    const std::function<std::string(EdgeType*)>& AddEdgeAttribute, StreamT& out_stream) const {
  out_stream << "digraph {\n";
  this->ForEachNode([&](NodeType* node) {
    if (IsNodeAllowed(node) == false) { return; }
    out_stream << "\"" << node->node_id_str() << "\" [label=\"" << node->VisualStr() << "\""
               << AddNodeAttribute(node) << "]\n";
  });
  this->ForEachEdge([&](EdgeType* edge) {
    if (IsEdgeAllowed(edge) == false) { return; }
    if (IsNodeAllowed(edge->src_node()) == false) { return; }
    if (IsNodeAllowed(edge->dst_node()) == false) { return; }
    out_stream << "\"" << edge->src_node()->node_id_str() << "\" -> "
               << "\"" << edge->dst_node()->node_id_str() << "\""
               << "[label=\"" << edge->VisualStr() << "\"" << AddEdgeAttribute(edge) << "];\n";
  });
  out_stream << "}\n";
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithFilePath(const std::string& file_path) const {
  auto log_stream = TeePersistentLogStream::Create(file_path);
  ToDotWithStream(log_stream);
  log_stream->Flush();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithFilePath(
    const std::function<std::string(NodeType*)>& AddNodeAttribute,
    const std::function<std::string(EdgeType*)>& AddEdgeAttribute,
    const std::string& file_path) const {
  auto log_stream = TeePersistentLogStream::Create(file_path);
  ToDotWithStream([](NodeType*) { return true; }, [](EdgeType*) { return true; }, AddNodeAttribute,
                  AddEdgeAttribute, log_stream);
  log_stream->Flush();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithFilePath(
    const std::function<bool(NodeType*)>& IsNodeAllowed,
    const std::function<bool(EdgeType*)>& IsEdgeAllowed, const std::string& file_path) const {
  auto log_stream = TeePersistentLogStream::Create(file_path);
  ToDotWithStream(
      IsNodeAllowed, IsEdgeAllowed, [](NodeType*) { return ""; }, [](EdgeType*) { return ""; },
      log_stream);
  log_stream->Flush();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ToDotWithAutoFilePath() const {
  std::string file_path = JoinPath("dot", TypeName(), NewUniqueId() + ".dot");
  ToDotWithFilePath(file_path);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::BfsForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
    const std::function<void(NodeType*)>& Handler) const {
  HashSet<NodeType*> queued_nodes;
  std::queue<NodeType*> queue;
  for (NodeType* start : starts) {
    if (queued_nodes.find(start) == queued_nodes.end()) {
      queue.push(start);
      queued_nodes.insert(start);
    }
  }
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    Handler(cur_node);
    ForEachNext(cur_node, [&](NodeType* next) {
      if (queued_nodes.find(next) == queued_nodes.end()) {
        queue.push(next);
        queued_nodes.insert(next);
      }
    });
  }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DfsForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
    const std::function<void(NodeType*)>& Handler) const {
  HashSet<NodeType*> visited_nodes;
  std::stack<NodeType*> stack;
  for (NodeType* start : starts) { stack.push(start); }
  while (!stack.empty()) {
    NodeType* cur_node = stack.top();
    stack.pop();
    if (visited_nodes.find(cur_node) == visited_nodes.end()) {
      Handler(cur_node);
      visited_nodes.insert(cur_node);
      ForEachNext(cur_node, [&](NodeType* next) {
        if (visited_nodes.find(next) == visited_nodes.end()) { stack.push(next); }
      });
    }
  }
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::FfsForEachNode(
    const std::function<void(const std::function<void(NodeType*)>&)>& ForEachStart,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachNext,
    const std::function<void(NodeType*)>& Handler) const {
  HashSet<NodeType*> visited_nodes;
  HashSet<NodeType*> handled_nodes;
  ForEachStart([&](NodeType* start) {
    if (visited_nodes.find(start) != visited_nodes.end()) { return; }
    std::stack<std::queue<NodeType*>> stack;
    stack.emplace(std::queue<NodeType*>{});
    stack.top().push(start);
    while (!stack.empty()) {
      if (stack.top().empty()) {
        stack.pop();
        continue;
      }
      if (handled_nodes.find(stack.top().front()) != handled_nodes.end()) {
        stack.top().pop();
        continue;
      }
      NodeType* cur_node = stack.top().front();
      if (visited_nodes.find(cur_node) == visited_nodes.end()) { visited_nodes.insert(cur_node); }
      int64_t next_unvisited_cnt = 0;
      ForEachNext(cur_node, [&](NodeType* next) {
        if (visited_nodes.find(next) == visited_nodes.end()) {
          if (next_unvisited_cnt == 0) { stack.emplace(std::queue<NodeType*>()); }
          stack.top().push(next);
          ++next_unvisited_cnt;
        }
      });
      if (next_unvisited_cnt == 0) {
        Handler(cur_node);
        handled_nodes.insert(cur_node);
      }
    }
  });
}

template<typename NodeType, typename EdgeType>
std::unique_ptr<HashSet<NodeType*>> Graph<NodeType, EdgeType>::FindFirstNontrivialSCC(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
    const {
  auto ForEachStart = [&](const std::function<void(NodeType*)>& Handler) {
    for (NodeType* start : starts) { Handler(start); }
  };
  return FindFirstNontrivialSCC(ForEachStart, ForEachInNode, ForEachOutNode);
}

template<typename NodeType, typename EdgeType>
std::unique_ptr<HashSet<NodeType*>> Graph<NodeType, EdgeType>::FindFirstNontrivialSCC(
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
    const {
  return FindFirstNontrivialSCC(
      [&](const std::function<void(NodeType*)>& Handler) { ForEachNode(Handler); }, ForEachInNode,
      ForEachOutNode);
}

template<typename NodeType, typename EdgeType>
std::unique_ptr<HashSet<NodeType*>> Graph<NodeType, EdgeType>::FindFirstNontrivialSCC() const {
  return FindFirstNontrivialSCC(
      [&](const std::function<void(NodeType*)>& Handler) { ForEachNode(Handler); },
      &NodeType::ForEachNodeOnInEdge, &NodeType::ForEachNodeOnOutEdge);
}

template<typename NodeType, typename EdgeType>
std::unique_ptr<HashSet<NodeType*>> Graph<NodeType, EdgeType>::FindFirstNontrivialSCC(
    const std::function<void(const std::function<void(NodeType*)>&)>& ForEachStart,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
    const {
  std::stack<NodeType*> stack;
  FfsForEachNode(ForEachStart, ForEachOutNode, [&](NodeType* node) { stack.push(node); });
  HashSet<NodeType*> visited;
  auto ForEachUnvisitedInNode = [&](NodeType* node, const std::function<void(NodeType*)>& Handler) {
    ForEachInNode(node, [&](NodeType* in_node) {
      if (visited.find(in_node) == visited.end()) { Handler(in_node); }
    });
  };
  while (stack.empty() == false) {
    NodeType* cur_node = stack.top();
    stack.pop();
    auto ret = std::make_unique<HashSet<NodeType*>>();
    DfsForEachNode({cur_node}, ForEachUnvisitedInNode,
                   [&](NodeType* node) { CHECK(ret->insert(node).second); });
    for (const auto& node : *ret) { visited.insert(node); }
    if (ret->size() > 1) { return ret; }
  }
  return std::unique_ptr<HashSet<NodeType*>>();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNodeDynamic(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  CHECK_JUST(TopoForEachNodeDynamicWithErrorCaptured(starts, ForEachInNode, ForEachOutNode,
                                                     [&](NodeType* node) {
                                                       Handler(node);
                                                       return Maybe<void>::Ok();
                                                     }));
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  CHECK_JUST(
      TopoForEachNodeWithErrorCaptured(starts, ForEachInNode, ForEachOutNode, [&](NodeType* node) {
        Handler(node);
        return Maybe<void>::Ok();
      }));
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::TopoForEachNode(
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  CHECK_JUST(TopoForEachNodeWithErrorCaptured(ForEachInNode, ForEachOutNode, [&](NodeType* node) {
    Handler(node);
    return Maybe<void>::Ok();
  }));
}

template<typename NodeType, typename EdgeType>
Maybe<void> Graph<NodeType, EdgeType>::TopoForEachNodeDynamicWithErrorCaptured(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<Maybe<void>(NodeType*)>& Handler) const {
  HashMap<NodeType*, bool> has_queued;
  std::queue<NodeType*> queue;
  for (NodeType* start : starts) {
    queue.push(start);
    has_queued[start] = true;
    ForEachInNode(start, [&](NodeType*) { LOG(FATAL) << "not a source"; });
  }
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    JUST(Handler(cur_node));
    ForEachOutNode(cur_node, [&](NodeType* out) {
      bool is_ready = true;
      ForEachInNode(out, [&](NodeType* in) {
        if (is_ready && !has_queued[in]) { is_ready = false; }
      });
      if (is_ready && !has_queued[out]) {
        queue.push(out);
        has_queued[out] = true;
      }
    });
  }
  return Maybe<void>::Ok();
}

template<typename NodeType, typename EdgeType>
Maybe<void> Graph<NodeType, EdgeType>::TopoForEachNodeWithErrorCaptured(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<Maybe<void>(NodeType*)>& Handler) const {
  HashMap<NodeType*, int32_t> counter_in;
  std::queue<NodeType*> queue;
  for (NodeType* start : starts) {
    queue.push(start);
    counter_in[start] = 0;
    ForEachInNode(start, [&](NodeType*) { LOG(FATAL) << "not a source"; });
  }
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    JUST(Handler(cur_node));
    ForEachOutNode(cur_node, [&](NodeType* out) {
      auto it = counter_in.find(out);
      // Move the initialization here
      if (it == counter_in.end()) {
        int32_t count = 0;
        ForEachInNode(out, [&](NodeType* out_in) { count++; });
        counter_in[out] = count;
        it = counter_in.find(out);
      }
      it->second--;
      if (it->second == 0) { queue.push(out); }
    });
  }
  return Maybe<void>::Ok();
}

template<typename NodeType, typename EdgeType>
Maybe<void> Graph<NodeType, EdgeType>::TopoForEachNodeWithErrorCaptured(
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<Maybe<void>(NodeType*)>& Handler) const {
  HashMap<NodeType*, int32_t> counter_in;
  std::queue<NodeType*> queue;
  ForEachNode([&](NodeType* node) {
    int32_t count = 0;
    ForEachInNode(node, [&](NodeType*) { count++; });
    counter_in[node] = count;
    if (count == 0) { queue.push(node); }
  });
  while (!queue.empty()) {
    NodeType* cur_node = queue.front();
    queue.pop();
    JUST(Handler(cur_node));
    ForEachOutNode(cur_node, [&](NodeType* out) {
      --counter_in[out];
      if (counter_in[out] == 0) { queue.push(out); }
    });
  }
  return Maybe<void>::Ok();
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DfsTopoForEachNodeSortByDistanceToSink(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  HashMap<NodeType*, int64_t> node2distance_to_sink;
  {
    std::list<NodeType*> nodes;
    TopoForEachNode(ForEachInNode, ForEachOutNode,
                    [&](NodeType* node) { nodes.emplace_back(node); });
    std::list<NodeType*> sinks;
    for (NodeType* node : nodes) {
      bool is_sink = true;
      ForEachOutNode(node, [&](NodeType* out_node) { is_sink = false; });
      if (is_sink) { sinks.emplace_back(node); }
    }
    TopoForEachNode(ForEachOutNode, ForEachInNode, [&](NodeType* node) {
      int64_t distance_to_sink = -1;
      ForEachOutNode(node, [&](NodeType* out_node) {
        distance_to_sink = std::max(distance_to_sink, node2distance_to_sink[out_node]);
      });
      node2distance_to_sink[node] = distance_to_sink + 1;
    });
  }
  auto ForEachOutNodeSortedByDistanceToSink = [&](NodeType* node,
                                                  const std::function<void(NodeType*)>& Handler) {
    std::vector<NodeType*> out_nodes;
    ForEachOutNode(node, [&](NodeType* out_node) { out_nodes.emplace_back(out_node); });
    std::sort(out_nodes.begin(), out_nodes.end(), [&](NodeType* lhs, NodeType* rhs) {
      // DfsTopoForEachNode use stack, so sort desc
      return node2distance_to_sink.at(lhs) > node2distance_to_sink.at(rhs);
    });
    for (NodeType* out_node : out_nodes) { Handler(out_node); }
  };
  DfsTopoForEachNode(starts, ForEachInNode, ForEachOutNodeSortedByDistanceToSink, Handler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::DfsTopoForEachNode(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode,
    const std::function<void(NodeType*)>& Handler) const {
  HashMap<NodeType*, bool> be_visited;
  std::stack<NodeType*> stack;
  for (NodeType* start : starts) {
    stack.push(start);
    ForEachInNode(start, [&](NodeType*) { LOG(FATAL) << "not a source"; });
  }
  while (!stack.empty()) {
    NodeType* cur_node = stack.top();
    stack.pop();
    Handler(cur_node);
    be_visited[cur_node] = true;
    ForEachOutNode(cur_node, [&](NodeType* out) {
      bool is_ready = true;
      ForEachInNode(out, [&](NodeType* in) {
        if (is_ready && !be_visited[in]) { is_ready = false; }
      });
      if (is_ready && !be_visited[out]) { stack.push(out); }
    });
  }
}

template<typename NodeType, typename EdgeType>
std::function<bool(const NodeType* src, const NodeType* dst)>
Graph<NodeType, EdgeType>::MakePredicatorIsReachable() const {
  return MakePredicatorIsReachable(source_nodes(), &NodeType::ForEachNodeOnInEdge,
                                   &NodeType::ForEachNodeOnOutEdge);
}

template<typename NodeType, typename EdgeType>
std::function<bool(const NodeType* src, const NodeType* dst)>
Graph<NodeType, EdgeType>::MakePredicatorIsReachable(
    const std::list<NodeType*>& starts,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachInNode,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachOutNode)
    const {
  static constexpr int64_t BITSET_SIZE = 512;  // size of cache line
  class BitSet {
   public:
    BitSet() = default;
    ~BitSet() = default;

    void Insert(int64_t k) { bitset_vec_.at(k / BITSET_SIZE).set(k % BITSET_SIZE, true); }

    bool Contains(int64_t k) { return bitset_vec_.at(k / BITSET_SIZE).test(k % BITSET_SIZE); }

    void Merge(const BitSet& other) {
      CHECK_EQ(bitset_vec_.size(), other.bitset_vec_.size());
      for (int64_t i = 0; i < bitset_vec_.size(); ++i) {
        bitset_vec_.at(i) |= other.bitset_vec_.at(i);
      }
    }

    void Resize(size_t size) {
      const int64_t bitset_vec_size = RoundUp(size, BITSET_SIZE) / BITSET_SIZE;
      bitset_vec_.resize(bitset_vec_size);
    }

   private:
    using bitset_vec = std::vector<std::bitset<BITSET_SIZE>>;
    bitset_vec bitset_vec_;
  };

  using NodePtr2Id = HashMap<const NodeType*, int64_t>;
  using Id2Ancestor = std::vector<BitSet>;
  std::shared_ptr<NodePtr2Id> node2id(new NodePtr2Id);
  std::shared_ptr<Id2Ancestor> id2ancestor(new Id2Ancestor(node_num()));
  int64_t id = 0;
  node2id->reserve(node_num());
  TopoForEachNode(ForEachInNode, ForEachOutNode, [&](NodeType* node) {
    node2id->emplace(node, id);
    id2ancestor->at(id).Resize(node_num());
    id += 1;
  });
  TopoForEachNode(ForEachInNode, ForEachOutNode, [&](NodeType* node) {
    const int64_t node_id = node2id->at(node);
    auto& ancestor_bitset_vec = id2ancestor->at(node_id);
    ForEachInNode(node, [&](NodeType* in_node) {
      const int64_t in_node_id = node2id->at(in_node);
      ancestor_bitset_vec.Insert(in_node_id);
      ancestor_bitset_vec.Merge(id2ancestor->at(in_node_id));
    });
  });
  return [id2ancestor, node2id](const NodeType* src, const NodeType* dst) -> bool {
    const int64_t dst_id = node2id->at(dst);
    return id2ancestor->at(dst_id).Contains(node2id->at(src));
  };
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachConnectedComponent(
    const std::function<void(const HashSet<NodeType*>&)>& Handler) const {
  ForEachConnectedComponent(
      [&](const std::function<void(NodeType*)>& Handler) { ForEachNode(Handler); },
      &NodeType::ForEachNodeOnInOutEdge, Handler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachConnectedComponent(
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
    const std::function<void(const HashSet<NodeType*>&)>& Handler) const {
  ForEachConnectedComponent(
      [&](const std::function<void(NodeType*)>& Handler) { ForEachNode(Handler); },
      ForEachConnected, Handler);
}

template<typename NodeType, typename EdgeType>
void Graph<NodeType, EdgeType>::ForEachConnectedComponent(
    const std::function<void(const std::function<void(NodeType*)>&)>& ForEachNodeAsStart,
    const std::function<void(NodeType*, const std::function<void(NodeType*)>&)>& ForEachConnected,
    const std::function<void(const HashSet<NodeType*>&)>& Handler) const {
  HashMap<NodeType*, int32_t> node2component_id;
  int32_t cur_component_id = 0;
  ForEachNodeAsStart([&](NodeType* start) {
    if (node2component_id.find(start) != node2component_id.end()) { return; }
    ++cur_component_id;
    BfsForEachNode({start}, ForEachConnected, [&](NodeType* node) {
      CHECK(node2component_id.emplace(node, cur_component_id).second);
    });
  });
  HashMap<int32_t, HashSet<NodeType*>> component_id2nodes;
  for (const auto& pair : node2component_id) { component_id2nodes[pair.second].insert(pair.first); }
  for (const auto& pair : component_id2nodes) { Handler(pair.second); }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_GRAPH_H_
