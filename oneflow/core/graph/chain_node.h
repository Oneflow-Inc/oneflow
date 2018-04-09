#ifndef ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_NODE_H_

#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ChainEdge;
class TaskGraph;

using CompTaskNodeHandler = std::function<void(CompTaskNode*)>;
using BldSubTskGphMthd = void (TaskGraph::*)(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box,
    std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);

using BldBoxingOpConfMthd = void (BoxingTaskNode::*)(
    const std::string& lbn,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const ChainNode* in_chain,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    const ChainNode* out_chain, BoxingOpConf*);

#define CHAIN_TYPE_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(Forward)      \
  OF_PP_MAKE_TUPLE_SEQ(Backward)     \
  OF_PP_MAKE_TUPLE_SEQ(RecordLoad)   \
  OF_PP_MAKE_TUPLE_SEQ(Decode)       \
  OF_PP_MAKE_TUPLE_SEQ(Loss)         \
  OF_PP_MAKE_TUPLE_SEQ(LossAcc)      \
  OF_PP_MAKE_TUPLE_SEQ(LossPrint)    \
  OF_PP_MAKE_TUPLE_SEQ(NormalMdUpdt) \
  OF_PP_MAKE_TUPLE_SEQ(MdSave)       \
  OF_PP_MAKE_TUPLE_SEQ(MdDiffAcc)    \
  OF_PP_MAKE_TUPLE_SEQ(Print)

class ChainNode : public Node<ChainNode, ChainEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainNode);
  virtual ~ChainNode() = default;

  // op_vec_
  std::shared_ptr<const Operator> SoleOp() const;
  const std::vector<std::shared_ptr<Operator>>& op_vec() const;
  std::vector<std::shared_ptr<Operator>>& mut_op_vec() { return op_vec_; }
  bool HasSoleRecurrentOp() const;

  // parallel_desc_
  std::shared_ptr<const ParallelDesc> parallel_desc() const;
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc();

  // data_output_lbns_
  const HashSet<std::string>& data_output_lbns() const {
    return data_output_lbns_;
  }
  virtual void set_data_output_lbns() {}

  // util
  virtual const char* TypeName() const = 0;
  std::string VisualStr() const;
  bool HasOpWithModelOrModelTmpBlob() const;
  bool HasOpWithModelBlob() const;
  bool HasOpWithForwardModelBlob() const;
  void GenSortedCompTaskNodes(
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId,
      CompTaskNodeHandler) const;
  int32_t GetModelSplitAxis() const;
  int32_t GetMaxModelSplitNum() const;

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
  virtual CompTaskNode* NewCompTaskNode() const {
    return NewCompTaskNodeWithSameName();
  }
  virtual CompTaskNode* NewCompTaskNodeWithSameName() const = 0;
  virtual void FixCompTaskNode(CompTaskNode*) const {}

  void AddDataOutputLbnsTo(const ChainNode*);

 private:
  bool HasOpWithCondition(std::function<bool(const Operator*)>) const;

  std::vector<std::shared_ptr<Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;

  HashSet<std::string> data_output_lbns_;
};

class BackwardChainNode;

#define OVERRIDE_PURE_VIRTUAL_METHOD()                                        \
  const char* TypeName() const override;                                      \
  BldSubTskGphMthd GetMthdForBldSubTskGphTo(const ChainNode*) const override; \
  BldBoxingOpConfMthd GetMthdForBldBoxingOpConfTo(const ChainNode*)           \
      const override;                                                         \
  std::vector<std::string> FindLbnsTo(const ChainNode*) const override;       \
  CompTaskNode* NewCompTaskNodeWithSameName() const override;

#define OVERRIDE_FROM_METHOD(x, y) x##From##y(const ChainNode*) const override;

#define CHAIN_NODE_BOILERPLATE(class_name) \
  OF_DISALLOW_COPY_AND_MOVE(class_name);   \
  class_name() = default;                  \
  ~class_name() = default;                 \
  OVERRIDE_PURE_VIRTUAL_METHOD();

class ForwardChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(ForwardChainNode);

  BackwardChainNode* bw_node() const { return bw_node_; }
  void set_bw_node(BackwardChainNode* val) { bw_node_ = val; }

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Forward)(Decode)(NormalMdUpdt));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Forward)(Decode));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Forward)(Decode));

  void set_data_output_lbns() override;

 private:
  CompTaskNode* NewCompTaskNode() const override;

  BackwardChainNode* bw_node_;
};

class BackwardChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(BackwardChainNode);

  ForwardChainNode* fw_node() const { return fw_node_; }
  void set_fw_node(ForwardChainNode* val) { fw_node_ = val; }

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Forward)(Backward)(Loss)(NormalMdUpdt));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Backward)(Loss));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Backward)(Loss));

  void set_data_output_lbns() override;

 private:
  CompTaskNode* NewCompTaskNode() const override;

  ForwardChainNode* fw_node_;
};

class RecordLoadChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(RecordLoadChainNode);

  void set_data_output_lbns() override {}
};

class DecodeChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(DecodeChainNode);

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (RecordLoad));

  void set_data_output_lbns() override;
};

class LossChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(LossChainNode);

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Forward)(Decode));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Forward)(Decode));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Forward)(Decode));

  void set_data_output_lbns() override;
};

class PrintChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintChainNode);
  PrintChainNode() = default;
  ~PrintChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Decode)(Forward)(Loss));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      OVERRIDE_FROM_METHOD, (BldBoxingOpConfMthd GetMthdForBldBoxingOpConf),
      (Decode)(Forward)(Loss));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (std::vector<std::string> FindLbns),
                                   (Decode)(Forward)(Loss));
};

class LossAccChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(LossAccChainNode);

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (Loss));
};

class LossPrintChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(LossPrintChainNode);

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

class NormalMdUpdtChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtChainNode);
  NormalMdUpdtChainNode() : random_seed_(NewRandomSeed()) {}
  ~NormalMdUpdtChainNode() = default;

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

 private:
  void FixCompTaskNode(CompTaskNode*) const override;

  uint32_t random_seed_;
};

class MdSaveChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(MdSaveChainNode);

  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD,
                                   (BldSubTskGphMthd GetMthdForBldSubTskGph),
                                   (NormalMdUpdt)(Forward));
};

class MdDiffAccChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(MdDiffAccChainNode);

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
