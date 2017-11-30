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
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);

using BldBoxingOpConfMthd = void (BoxingTaskNode::*)(
    const std::string& lbn,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    int64_t in_parallel_num,
    const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    int64_t out_parallel_num, BoxingOpConf*);

#define CHAIN_TYPE_SEQ            \
  OF_PP_MAKE_TUPLE_SEQ(Forward)   \
  OF_PP_MAKE_TUPLE_SEQ(Backward)  \
  OF_PP_MAKE_TUPLE_SEQ(Source)    \
  OF_PP_MAKE_TUPLE_SEQ(Loss)      \
  OF_PP_MAKE_TUPLE_SEQ(LossAcc)   \
  OF_PP_MAKE_TUPLE_SEQ(LossPrint) \
  OF_PP_MAKE_TUPLE_SEQ(MdUpdt)    \
  OF_PP_MAKE_TUPLE_SEQ(MdSave)    \
  OF_PP_MAKE_TUPLE_SEQ(MdDiffAcc) \
  OF_PP_MAKE_TUPLE_SEQ(Print)

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

  // data_output_lbns_
  const HashSet<std::string>& data_output_lbns() const {
    return data_output_lbns_;
  }
  virtual void set_data_output_lbns() {}

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
  virtual void FixCompTaskNode(CompTaskNode*) const {}

  void AddDataOutputLbnsTo(const ChainNode*);

 private:
  std::vector<std::shared_ptr<const Operator>> op_vec_;
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
  CompTaskNode* NewCompTaskNode() const override;

#define CHAIN_NODE_BOILERPLATE(class_name) \
  OF_DISALLOW_COPY_AND_MOVE(class_name);   \
  class_name() = default;                  \
  ~class_name() = default;                 \
  OVERRIDE_PURE_VIRTUAL_METHOD();

#define OVERRIDE_FROM_METHOD(ret_type, prefix, chain_type, return_value) \
  ret_type prefix##From##chain_type(const ChainNode*) const override;

#define MAKE_CHAIN_FUNC_SEQ(ret_type, prefix, t) \
  OF_PP_MAP_PREPEND(ret_type, OF_PP_MAP_PREPEND(prefix, OF_PP_TUPLE_TO_SEQ(t)))

class ForwardChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(ForwardChainNode);

  BackwardChainNode* bw_node() const { return bw_node_; }
  void set_bw_node(BackwardChainNode* val) { bw_node_ = val; }

#define FORWARD_CHAIN_NODE_FUNC_SEQ                                       \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph,           \
                      ((Forward, &TaskGraph::BldSubTskGphByBoxing),       \
                       (Source, &TaskGraph::BldSubTskGphByBoxing),        \
                       (MdUpdt, &TaskGraph::BldSubTskGphByOneToOne)))     \
  MAKE_CHAIN_FUNC_SEQ(                                                    \
      BldBoxingOpConfMthd, GetMthdForBldBoxingOpConf,                     \
      ((Forward, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this)), \
       (Source, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this)))) \
  MAKE_CHAIN_FUNC_SEQ(std::vector<std::string>, FindLbns,                 \
                      ((Forward, FindLbnsBetweenFw(node, this)),          \
                       (Source, FindLbnsBetweenFw(node, this))))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, FORWARD_CHAIN_NODE_FUNC_SEQ);

  void set_data_output_lbns() override;

 private:
  BackwardChainNode* bw_node_;
};

class BackwardChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(BackwardChainNode);

  ForwardChainNode* fw_node() const { return fw_node_; }
  void set_fw_node(ForwardChainNode* val) { fw_node_ = val; }

#define BACKWARD_CHAIN_NODE_FUNC_SEQ                                       \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph,            \
                      ((Forward, &TaskGraph::BldSubTskGphByOneToOne),      \
                       (Backward, &TaskGraph::BldSubTskGphByBoxing),       \
                       (Loss, &TaskGraph::BldSubTskGphByBoxing),           \
                       (MdUpdt, &TaskGraph::BldSubTskGphByOneToOne)))      \
  MAKE_CHAIN_FUNC_SEQ(                                                     \
      BldBoxingOpConfMthd, GetMthdForBldBoxingOpConf,                      \
      ((Backward, GetBldBoxingOpConfMethodByBwParallelPolicy(node, this)), \
       (Loss, GetBldBoxingOpConfMethodByBwParallelPolicy(node, this))))    \
  MAKE_CHAIN_FUNC_SEQ(std::vector<std::string>, FindLbns,                  \
                      ((Backward, FindLbnsBetweenBw(node, this)),          \
                       (Loss, FindLbnsBetweenBw(node, this))))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, BACKWARD_CHAIN_NODE_FUNC_SEQ);

  void set_data_output_lbns() override;

 private:
  ForwardChainNode* fw_node_;
};

class SourceChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(SourceChainNode);

  void set_data_output_lbns() override;
};

class LossChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(LossChainNode);

#define LOSS_CHAIN_NODE_FUNC_SEQ                                          \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph,           \
                      ((Forward, &TaskGraph::BldSubTskGphByBoxing),       \
                       (Source, &TaskGraph::BldSubTskGphByBoxing)))       \
  MAKE_CHAIN_FUNC_SEQ(                                                    \
      BldBoxingOpConfMthd, GetMthdForBldBoxingOpConf,                     \
      ((Forward, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this)), \
       (Source, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this)))) \
  MAKE_CHAIN_FUNC_SEQ(std::vector<std::string>, FindLbns,                 \
                      ((Forward, FindLbnsBetweenFw(node, this)),          \
                       (Source, FindLbnsBetweenFw(node, this))))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, LOSS_CHAIN_NODE_FUNC_SEQ);
};

class PrintChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(PrintChainNode);

#define PRINT_CHAIN_NODE_FUNC_SEQ                                         \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph,           \
                      ((Source, &TaskGraph::BldSubTskGphByBoxing),        \
                       (Forward, &TaskGraph::BldSubTskGphByBoxing),       \
                       (Loss, &TaskGraph::BldSubTskGphByBoxing)))         \
  MAKE_CHAIN_FUNC_SEQ(                                                    \
      BldBoxingOpConfMthd, GetMthdForBldBoxingOpConf,                     \
      ((Source, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this)),  \
       (Forward, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this)), \
       (Loss, GetBldBoxingOpConfMethodByFwParallelPolicy(node, this))))   \
  MAKE_CHAIN_FUNC_SEQ(std::vector<std::string>, FindLbns,                 \
                      ((Source, FindLbnsBetweenFw(node, this)),           \
                       (Forward, FindLbnsBetweenFw(node, this)),          \
                       (Loss, FindLbnsBetweenFw(node, this))))
  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, PRINT_CHAIN_NODE_FUNC_SEQ);
};

class LossAccChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(LossAccChainNode);

#define LOSS_ACC_CHAIN_NODE_FUNC_SEQ                            \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph, \
                      ((Loss, &TaskGraph::BldSubTskGphByOneToOne)))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, LOSS_ACC_CHAIN_NODE_FUNC_SEQ);
};

class LossPrintChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(LossPrintChainNode);

#define LOSS_PRINT_CHAIN_NODE_FUNC_SEQ                               \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph,      \
                      ((LossAcc, &TaskGraph::BldSubTskGphByBoxing))) \
  MAKE_CHAIN_FUNC_SEQ(                                               \
      BldBoxingOpConfMthd, GetMthdForBldBoxingOpConf,                \
      ((LossAcc, &BoxingTaskNode::BldBoxingOpConfWithAddAndClone)))  \
  MAKE_CHAIN_FUNC_SEQ(std::vector<std::string>, FindLbns,            \
                      ((LossAcc, {kPackedBlobName})))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, LOSS_PRINT_CHAIN_NODE_FUNC_SEQ);
};

class MdUpdtChainNode final : public ChainNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtChainNode);
  MdUpdtChainNode() : random_seed_(NewRandomSeed()) {}
  ~MdUpdtChainNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();

#define MDUPDT_CHAIN_NODE_FUNC_SEQ                                    \
  MAKE_CHAIN_FUNC_SEQ(                                                \
      BldSubTskGphMthd, GetMthdForBldSubTskGph,                       \
      ((MdDiffAcc, GetMthdForBldSubTskGphByParellPolicy(this))))      \
  MAKE_CHAIN_FUNC_SEQ(                                                \
      BldBoxingOpConfMthd, GetMthdForBldBoxingOpConf,                 \
      ((MdDiffAcc, &BoxingTaskNode::BldBoxingOpConfWithAddAndClone))) \
  MAKE_CHAIN_FUNC_SEQ(std::vector<std::string>, FindLbns,             \
                      ((MdDiffAcc, {kPackedBlobName})))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, MDUPDT_CHAIN_NODE_FUNC_SEQ);

 private:
  void FixCompTaskNode(CompTaskNode*) const override;

  uint32_t random_seed_;
};

class MdSaveChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(MdSaveChainNode);

#define MDSAVE_CHAIN_NODE_FUNC_SEQ                              \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph, \
                      ((MdUpdt, GetMthdForBldSubTskGphByParallelNum(this))))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, MDSAVE_CHAIN_NODE_FUNC_SEQ);
};

class MdDiffAccChainNode final : public ChainNode {
 public:
  CHAIN_NODE_BOILERPLATE(MdDiffAccChainNode);

#define MDDIFF_ACC_CHAIN_NODE_FUNC_SEQ                          \
  MAKE_CHAIN_FUNC_SEQ(BldSubTskGphMthd, GetMthdForBldSubTskGph, \
                      ((Backward, &TaskGraph::BldSubTskGphByOneToOne)))

  OF_PP_FOR_EACH_TUPLE(OVERRIDE_FROM_METHOD, MDDIFF_ACC_CHAIN_NODE_FUNC_SEQ);
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
