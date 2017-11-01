#ifndef ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ChainEdge;
using CompTaskNodeHandler = std::function<void(CompTaskNode*)>;
class TaskGraph;
using BldSubTskGphMthd = void (TaskGraph::*)(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);

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
  void GenSortedCompTaskNodes(CompTaskNodeHandler) const;

  // Get method for build sub-taskgraph
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphTo(
      const ChainNode* dst_node) const = 0;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromFw() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromBw() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromSrc() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromLoss() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromLossAcc() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromLossRecord() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromMdUpdt() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromMdSave() const;
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFromMdDiffAcc() const;

 protected:
  ChainNode() = default;
  virtual CompTaskNode* NewCompTaskNode() const = 0;

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
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromFw() const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromSrc() const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromMdUpdt() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
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
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromFw() const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromLoss() const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromMdUpdt() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
  ForwardChainNode* fw_node_;
};

class SourceChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceChainNode);
  SourceChainNode() = default;
  ~SourceChainNode() = default;

  virtual const char* TypeName() const { return "SourceChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class LossChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossChainNode);
  LossChainNode() = default;
  ~LossChainNode() = default;

  virtual const char* TypeName() const { return "LossChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromFw() const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromSrc() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class LossAccChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccChainNode);
  LossAccChainNode() = default;
  ~LossAccChainNode() = default;

  virtual const char* TypeName() const { return "LossAccChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromLoss() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class LossRecordChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordChainNode);
  LossRecordChainNode() = default;
  ~LossRecordChainNode() = default;

  virtual const char* TypeName() const { return "LossRecordChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromLossAcc() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class MdUpdtChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtChainNode);
  MdUpdtChainNode() = default;
  ~MdUpdtChainNode() = default;

  virtual const char* TypeName() const { return "MdUpdtChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromMdDiffAcc() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class MdSaveChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveChainNode);
  MdSaveChainNode() = default;
  ~MdSaveChainNode() = default;

  virtual const char* TypeName() const { return "MdSaveChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromMdUpdt() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class MdDiffAccChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccChainNode);
  MdDiffAccChainNode() = default;
  ~MdDiffAccChainNode() = default;

  virtual const char* TypeName() const { return "MdDiffAccChainNode"; }
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override;
  BldSubTskGphMthd GetMthdForBldSubTskGphFromBw() const override;

 private:
  CompTaskNode* NewCompTaskNode() const override;
};

class ChainEdge final : public Edge<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainEdge);
  ChainEdge() = default;
  ~ChainEdge() = default;

  std::string VisualStr() const override;

  BldSubTskGphMthd GetMthdForBldSubTskGph() const;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_
