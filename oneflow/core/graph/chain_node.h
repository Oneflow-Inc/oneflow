#ifndef ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_

#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ChainEdge;
class TaskGraph;

using CompTaskNodeHandler = std::function<void(CompTaskNode*)>;
using BldSubTskGphMthd = void (TaskGraph::*)(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);

using BldBoxingOpConfMthd = void (BoxingTaskNode::*)(
    const std::string& lbn,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    int64_t in_parallel_num, int64_t in_edge_first, int64_t in_edge_last,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t out_parallel_num, int64_t* used_out_edge_begin, BoxingOpConf*);

#define CHAIN_TYPE_SEQ             \
  OF_PP_MAKE_TUPLE_SEQ(Forward)    \
  OF_PP_MAKE_TUPLE_SEQ(Backward)   \
  OF_PP_MAKE_TUPLE_SEQ(Source)     \
  OF_PP_MAKE_TUPLE_SEQ(Loss)       \
  OF_PP_MAKE_TUPLE_SEQ(LossAcc)    \
  OF_PP_MAKE_TUPLE_SEQ(LossRecord) \
  OF_PP_MAKE_TUPLE_SEQ(MdUpdt)     \
  OF_PP_MAKE_TUPLE_SEQ(MdSave)     \
  OF_PP_MAKE_TUPLE_SEQ(MdDiffAcc)

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

  // util
  virtual const char* TypeName() const = 0;
  std::string VisualStr() const;
  bool HasOpWithModelOrModelTmpBlob() const;
  void GenSortedCompTaskNodes(CompTaskNodeHandler) const;

  // To
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const = 0;
  virtual BldBoxingOpConfMthd GetMthdForBldBoxingOpConfTo(
      const ChainNode*) const = 0;
  virtual std::vector<std::string> FindLbnsTo(const ChainNode*) const = 0;

// From
#define DECLARE_VIRTUAL_FROM_METHOD(x)                                     \
  virtual BldSubTskGphMthd GetMthdForBldSubTskGphFrom##x(const ChainNode*) \
      const;                                                               \
  virtual BldBoxingOpConfMthd GetMthdForBldBoxingOpConfFrom##x(            \
      const ChainNode*) const;                                             \
  virtual std::vector<std::string> FindLbnsFrom##x(const ChainNode*) const;

  OF_PP_FOR_EACH_TUPLE(DECLARE_VIRTUAL_FROM_METHOD, CHAIN_TYPE_SEQ);
#undef DECLARE_VIRTUAL_METHOD

 protected:
  ChainNode() = default;
  virtual CompTaskNode* NewCompTaskNode() const = 0;

 private:
  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class BackwardChainNode;

#define OVERRIDE_PURE_VIRTUAL_METHOD()                                        \
  const char* TypeName() const override;                                      \
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override; \
  BldBoxingOpConfMthd GetMthdForBldBoxingOpConfTo(const ChainNode*)           \
      const override;                                                         \
  std::vector<std::string> FindLbnsTo(const ChainNode*) const override;       \
  CompTaskNode* NewCompTaskNode() const override;

#define OVERRIDE_FROM_METHOD(x, y) x##From##y(const ChainNode*) const override;

class ForwardChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardChainNode);
  ForwardChainNode() = default;
  ~ForwardChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  BackwardChainNode* bw_node() const { return bw_node_; }
  void set_bw_node(BackwardChainNode* val) { bw_node_ = val; }

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Forward)(Source)(MdUpdt));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Forward)(Source));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Forward)(Source));

 private:
  BackwardChainNode* bw_node_;
};

class BackwardChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardChainNode);
  BackwardChainNode() = default;
  ~BackwardChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  ForwardChainNode* fw_node() const { return fw_node_; }
  void set_fw_node(ForwardChainNode* val) { fw_node_ = val; }

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Forward)(Backward)(Loss)(MdUpdt));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Backward)(Loss));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Backward)(Loss));

 private:
  ForwardChainNode* fw_node_;
};

class SourceChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceChainNode);
  SourceChainNode() = default;
  ~SourceChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();
};

class LossChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossChainNode);
  LossChainNode() = default;
  ~LossChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Forward)(Source));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Forward)(Source));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Forward)(Source));
};

class LossAccChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossAccChainNode);
  LossAccChainNode() = default;
  ~LossAccChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Loss));
};

class LossRecordChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordChainNode);
  LossRecordChainNode() = default;
  ~LossRecordChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (LossAcc));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (LossAcc));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (LossAcc));
};

class MdUpdtChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtChainNode);
  MdUpdtChainNode() = default;
  ~MdUpdtChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (MdDiffAcc));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (MdDiffAcc));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (MdDiffAcc));
};

class MdSaveChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveChainNode);
  MdSaveChainNode() = default;
  ~MdSaveChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (MdUpdt));
};

class MdDiffAccChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccChainNode);
  MdDiffAccChainNode() = default;
  ~MdDiffAccChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Backward));
};

std::vector<std::string> FindLbnsBetween(const ChainNode* in_chain,
                                         const ChainNode* out_chain);

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
