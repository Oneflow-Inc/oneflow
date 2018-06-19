#ifndef ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_

#include "oneflow/core/graph/graph.h"

namespace oneflow {

class TaskNode;
class ChainNode;

struct Chain {
  // nodes belong to this chain
  std::vector<TaskNode*> nodes;
  // ancestors of the nodes in this chain
  HashSet<TaskNode*> ancestors;
  // ancestors_and_this = nodes + ancestors
  HashSet<TaskNode*> ancestors_and_this;
  // descendants of the nodes in this chain
  HashSet<TaskNode*> descendants;
  // descendants_and_this = nodes + descendants
  HashSet<TaskNode*> descendants_and_this;
  int64_t stream_id;
  int64_t area_id;
  ChainNode* chain_node;
};

using ChainIt = std::list<Chain>::iterator;
using Task2ChainItMap = HashMap<const TaskNode*, ChainIt>;

class ChainEdge;

class ChainNode final : public Node<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainNode);
  explicit ChainNode(ChainIt chain_it) : chain_it_(chain_it) {}
  virtual ~ChainNode() = default;

  std::string VisualStr() const override;

 private:
  ChainIt chain_it_;
};

class ChainEdge final : public Edge<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainEdge);
  ChainEdge() = default;
  ~ChainEdge() = default;

 private:
};

class TaskGraph;

class ChainGraph final : public Graph<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainGraph);
  ChainGraph() = delete;
  ~ChainGraph() = default;

  ChainGraph(const TaskGraph& task_gph);
  const char* TypeName() const override { return "ChainGraph"; }

 private:
  ChainNode* ChainNode4TaskNode(TaskNode* task_node) const;
  const TaskGraph& task_gph_;
  std::list<Chain> chain_list_;
  Task2ChainItMap task_node2chain_it_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
