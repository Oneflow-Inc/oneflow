#ifndef ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include <bitset>

namespace oneflow {

class TaskNode;
class ChainNode;

struct Chain {
  // nodes belong to this chain
  std::vector<TaskNode*> nodes;
  // ancestors of the nodes in this chain
  std::bitset<MAX_TASK_NODE_NUM> ancestors;
  // ancestors_and_this = nodes + ancestors
  std::bitset<MAX_TASK_NODE_NUM> ancestors_and_this;
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
  explicit ChainNode(ChainIt chain_it) : chain_it_(chain_it), chain_id_(-1) {}
  virtual ~ChainNode() = default;

  std::string VisualStr() const override;
  ChainIt chain_it() const { return chain_it_; }
  const std::vector<TaskNode*>& ordered_task_nodes() const { return chain_it_->nodes; }
  int64_t chain_id() const {
    CHECK_NE(chain_id_, -1);
    return chain_id_;
  }
  void set_chain_id(int64_t val) { chain_id_ = val; }

 private:
  ChainIt chain_it_;
  int64_t chain_id_;
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
  const std::vector<ChainNode*>& ordered_chain_nodes() const { return ordered_chain_nodes_; }
  const std::vector<TaskNode*>& ordered_task_nodes() const { return ordered_task_nodes_; }

 private:
  ChainNode* ChainNode4TaskNode(TaskNode* task_node) const;
  bool HasChainEdge(ChainNode* src, ChainNode* dst) const;
  const TaskGraph& task_gph_;
  std::list<Chain> chain_list_;
  Task2ChainItMap task_node2chain_it_;
  std::vector<ChainNode*> ordered_chain_nodes_;
  std::vector<TaskNode*> ordered_task_nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
