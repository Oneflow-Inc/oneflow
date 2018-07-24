#ifndef ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include <bitset>

namespace oneflow {
const int64_t MAX_ANCESTOR_NUM = 10000;

class TaskNode;
class ChainNode;

struct Chain {
  // nodes belong to this chain
  std::vector<TaskNode*> nodes;
  // ancestors of the nodes in this chain
  std::bitset<MAX_ANCESTOR_NUM> ancestors;
  // ancestors_and_this = nodes + ancestors
  std::bitset<MAX_ANCESTOR_NUM> ancestors_and_this;
  int64_t stream_id;
  int64_t area_id;
};

using ChainIt = std::list<Chain>::iterator;

class ChainEdge;

class ChainNode final : public Node<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainNode);
  explicit ChainNode(const std::vector<TaskNode*>& task_nodes)
      : task_nodes_(task_nodes), chain_id_(-1) {}
  virtual ~ChainNode() = default;

  std::string VisualStr() const override;
  const std::vector<TaskNode*>& task_nodes() const { return task_nodes_; }
  int64_t chain_id() const {
    CHECK_NE(chain_id_, -1);
    return chain_id_;
  }
  void set_chain_id(int64_t val) { chain_id_ = val; }

 private:
  std::vector<TaskNode*> task_nodes_;
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
  void GroupTaskNodesByMachine(const std::vector<TaskNode*>& ordered_task_nodes,
                               HashMap<int64_t, std::vector<TaskNode*>>* machine2tasks);
  void MergeTaskNodes(const HashMap<int64_t, std::vector<TaskNode*>>& machine2tasks,
                      std::vector<std::vector<TaskNode*>>* chains);

  const TaskGraph& task_gph_;
  HashMap<TaskNode*, ChainNode*> task_node2chain_node_;
  std::vector<ChainNode*> ordered_chain_nodes_;
  std::vector<TaskNode*> ordered_task_nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
