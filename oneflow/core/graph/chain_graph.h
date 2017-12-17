#ifndef ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_

#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ChainGraph final : public Graph<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainGraph);
  ChainGraph() = default;
  ~ChainGraph() = default;

  ChainGraph(bool is_train);

  const char* TypeName() const override { return "ChainGraph"; }

 private:
  template<typename ChainNodeType>
  void ForEachChainNode(std::function<void(ChainNodeType*)> Handler) {
    // the Handler may call "NewNode"
    std::vector<ChainNodeType*> valid_nodes;
    ForEachNode([&](ChainNode* chain_node) {
      auto valid_node = dynamic_cast<ChainNodeType*>(chain_node);
      if (valid_node != nullptr) { valid_nodes.push_back(valid_node); }
    });
    for (ChainNodeType* valid_node : valid_nodes) { Handler(valid_node); }
  }

  void BuildFwStruct();
  void BuildBwStruct();
  void BuildLossPrintStruct();
  void BuildModelStruct(bool is_train);
  void BuildRnnStruct();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
