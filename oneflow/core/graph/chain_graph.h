#ifndef ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_

#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ChainEdge;

class ChainNode : public Node<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainNode);
  virtual ~ChainNode() = default;

  // op_vec_
  std::shared_ptr<const Operator> SoleOp() const;
  const std::vector<std::shared_ptr<const Operator>>& op_vec() const;
  std::vector<std::shared_ptr<const Operator>>& mut_op_vec() { return op_vec_; }

  // parallel_desc_
  std::shared_ptr<const ParallelDesc> parallel_desc() const;
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc();

  // others
  virtual const char* TypeName() const = 0;
  std::string VisualStr() const;
  bool HasOpWithModelOrModelTmpBlob() const;

 protected:
  ChainNode() = default;

 private:
  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class BackwardChainNode;

class ForwardChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardChainNode);
  ForwardChainNode() = default;
  ~ForwardChainNode() = default;

  virtual const char* TypeName() const { return "ForwardChainNode"; }

  BackwardChainNode* bw_node() const { return bw_node_; }
  void set_bw_node(BackwardChainNode* val) { bw_node_ = val; }

 private:
  BackwardChainNode* bw_node_;
};

class BackwardChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardChainNode);
  BackwardChainNode() = default;
  ~BackwardChainNode() = default;

  virtual const char* TypeName() const { return "BackwardChainNode"; }

  ForwardChainNode* fw_node() const { return fw_node_; }
  void set_fw_node(ForwardChainNode* val) { fw_node_ = val; }

 private:
  ForwardChainNode* fw_node_;
};

class LossChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossChainNode);
  LossChainNode() = default;
  ~LossChainNode() = default;

  virtual const char* TypeName() const { return "LossChainNode"; }

 private:
};

class LossAccChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccChainNode);
  LossAccChainNode() = default;
  ~LossAccChainNode() = default;

  virtual const char* TypeName() const { return "LossAccChainNode"; }

 private:
};

class LossRecordChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordChainNode);
  LossRecordChainNode() = default;
  ~LossRecordChainNode() = default;

  virtual const char* TypeName() const { return "LossRecordChainNode"; }

 private:
};

class ChainEdge final : public Edge<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainEdge);
  ChainEdge() = default;
  ~ChainEdge() = default;

  std::string VisualStr() const override;

 private:
};

class ChainGraph final : public Graph<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainGraph);
  ChainGraph() = default;
  ~ChainGraph() = default;

  ChainGraph(const LogicalGraph& logical_gph, bool is_train);

  const char* TypeName() const override { return "ChainGraph"; }

 private:
  template<typename ChainNodeType>
  ChainNodeType* NewChainNode() {
    static_assert(std::is_base_of<ChainNode, ChainNodeType>::value, "");
    ChainNodeType* ret = new ChainNodeType;
    AddAllocatedNode(ret);
    return ret;
  }
  template<typename ChainNodeType>
  void ForEachChainNode(std::function<void(ChainNodeType*)> Handler) {
    // the Handler may call "NewChainNode"
    std::vector<ChainNodeType*> valid_nodes;
    ForEachNode([&](ChainNode* chain_node) {
      auto valid_node = dynamic_cast<ChainNodeType*>(chain_node);
      if (valid_node != nullptr) { valid_nodes.push_back(valid_node); }
    });
    for (ChainNodeType* valid_node : valid_nodes) { Handler(valid_node); }
  }

  void BuildFwStruct(const LogicalGraph& logical_gph);
  void BuildBwStruct();
  void BuildLossRecordStruct();
  void BuildModelStruct(bool is_train);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_GRAPH_H_
