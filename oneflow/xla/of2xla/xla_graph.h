#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_H_

#include <vector>
#include "oneflow/xla/of2xla/xla_argument.h"
#include "oneflow/xla/of2xla/xla_node.h"

namespace oneflow {
namespace mola {

class XlaGraph {
 public:
  XlaGraph() = default;
  explicit XlaGraph(const OpGraph *op_graph);

  virtual ~XlaGraph();

  XlaNode *Node(int64_t node_id);
  const XlaNode *Node(int64_t node_id) const;

  XlaNode *AddNode();
  XlaNode *AddNode(const Operator *op);
  XlaNode *AddArgumentNode(const XlaLaunchOpConf::Argument &arg_conf);
  XlaEdge *AddEdge(XlaNode *start, XlaNode *end);

  // Create a subgraph for node that unique id is `node_id`
  XlaGraph *AddSubGraph(int64_t node_id);

  const std::vector<XlaNode *> &Nodes() const { return nodes_; }
  std::vector<XlaNode *> &Nodes() { return nodes_; }

  XlaEdge *Connect(XlaNode *start, XlaNode *end);
  XlaEdge *Connect(XlaNode *start, XlaNode *end, const Argument &arg);
  void Disconnect(XlaEdge *edge);

  std::vector<Argument> Arguments() const;

 private:
  void BuildEdges();

  // All allocated nodes in the graph. The node unique id is related to it's
  // index in the vector. The Xla node in `nodes_` can be nullptr since we will
  // always keep it in `nodes_` even if it has been removed from the graph.
  std::vector<XlaNode *> nodes_;

  // All allocated edges in the graph. The edge unique id is related to it's
  // index in the vector. And the xla edge in `edges_` can also be nullptr.
  std::vector<XlaEdge *> edges_;

  // All allocated subgraphs. The key of the map means node unique id, and the
  // value is the subgraph which belongs to the node
  std::unordered_map<int64_t, XlaGraph *> subgraphs_;
};

class XlaLaunchGraph : public XlaGraph {
 public:
  explicit XlaLaunchGraph(const XlaLaunchOpConf &launch_conf,
                          const JobDesc *job_desc)
      : launch_conf_(launch_conf), job_desc_(job_desc) {
    SetupArguments();
    BuildLaunchGraph();
  }

  LogicalBlobId Input(const std::string &name) const {
    const auto &it = inputs_.find(name);
    DCHECK(it != inputs_.end());
    return it->second;
  }

  LogicalBlobId Output(const std::string &name) const {
    const auto &it = outputs_.find(name);
    DCHECK(it != outputs_.end());
    return it->second;
  }

 private:
  void SetupArguments();
  void BuildLaunchGraph();

  const XlaLaunchOpConf &launch_conf_;

  const JobDesc *job_desc_;

  std::vector<std::shared_ptr<Operator> > allocated_ops_;
  // "fc/out" --> "fc/out"
  std::unordered_map<std::string, LogicalBlobId> inputs_;
  // "out0" --> "fc/out"
  std::unordered_map<std::string, LogicalBlobId> outputs_;
  std::vector<XlaLaunchOpConf::Argument> argument_proto_;
};

template <typename GraphType>
struct GraphTrait {
  typedef typename GraphType::NodeType *pNodeType;
  typedef typename GraphType::EdgeType *pEdgeType;
};

template <>
struct GraphTrait<XlaGraph> {
  typedef XlaNode *pNodeType;
  typedef XlaEdge *pEdgeType;
};

template <>
struct GraphTrait<const XlaGraph> {
  typedef const XlaNode *pNodeType;
  typedef const XlaEdge *pEdgeType;
};

template <>
struct GraphTrait<XlaLaunchGraph> : public GraphTrait<XlaGraph> {};

template <>
struct GraphTrait<const XlaLaunchGraph> : public GraphTrait<const XlaGraph> {};

template <typename GraphType, typename UserFunc>
void TopologyVisit(GraphType &graph, UserFunc func) {
  typedef typename GraphTrait<GraphType>::pNodeType pNodeType;
  typedef typename GraphTrait<GraphType>::pEdgeType pEdgeType;

  std::unordered_set<pNodeType> visited;
  std::queue<pNodeType> visit_queue;
  for (pNodeType node : graph.Nodes()) {
    if (node->IsSourceNode()) {
      visit_queue.push(node);
      visited.insert(node);
    }
  }

  auto IsAllInputsVisited = [&](pNodeType node) -> bool {
    for (pEdgeType edge : node->in_edges()) {
      pNodeType start = edge->start();
      if (visited.count(start) == 0) {
        return false;
      }
    }
    return true;
  };

  while (!visit_queue.empty()) {
    pNodeType node = visit_queue.front();
    visit_queue.pop();
    { // Run user function
      func(node);
    }
    for (pEdgeType edge : node->out_edges()) {
      pNodeType end = edge->end();
      if (IsAllInputsVisited(end) &&
          visited.insert(end).second) {
        visit_queue.push(end);
      }
    }
  }
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_GRAPH_NODE_H_
