#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_

#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/reduce_rank_context.h"
#include "oneflow/core/graph/pack_forward_task_node.h"
#include "oneflow/core/graph/unpack_forward_task_node.h"
#include "oneflow/core/graph/unpack_backward_task_node.h"
#include "oneflow/core/graph/repeat_forward_compute_task_node.h"
#include "oneflow/core/graph/repeat_backward_compute_task_node.h"

namespace oneflow {

class LogicalEdge;

class LogicalNode : public Node<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalNode);
  virtual ~LogicalNode() = default;

  // op_vec_
  std::shared_ptr<Operator> SoleOp() const;
  const std::vector<std::shared_ptr<Operator>>& op_vec() const { return op_vec_; }
  std::vector<std::shared_ptr<Operator>>& mut_op_vec() { return op_vec_; }

  // parallel_desc_
  std::shared_ptr<const ParallelDesc> parallel_desc() const { return parallel_desc_; }
  std::shared_ptr<const ParallelDesc>& mut_parallel_desc() { return parallel_desc_; }

  // shared_model_nodes_
  std::shared_ptr<const std::vector<LogicalNode*>> shared_model_nodes() const {
    return shared_model_nodes_;
  }
  std::shared_ptr<const std::vector<LogicalNode*>>& mut_shared_model_nodes() {
    return shared_model_nodes_;
  }

  // Lbis
  std::vector<LogicalBlobId> GetLbisTo(const LogicalNode* dst) const;
  void SetDataLbisTo(const LogicalNode* dst, const std::vector<LogicalBlobId>&);
  bool IsDataLbiOnOutEdge(const LogicalBlobId& lbi) const;

  // util
  virtual std::string TypeName() const = 0;
  std::string VisualStr() const;
  bool HasOpWithModelOrConstModelBlob() const;
  bool HasOpWithModelBlob() const;
  bool HasOpWithForwardModelBlob() const;
  void GenSortedCompTaskNodes(std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly,
                              std::vector<std::pair<int64_t, CompTaskNode*>>* nodes,
                              std::function<void(CompTaskNode*)>) const;

  // model split
  LogicalNode* main_model_parallel() const { return main_model_parallel_; }
  void set_main_model_parallel(LogicalNode* val) { main_model_parallel_ = val; }
  int32_t GetModelSplitAxis() const;
  int32_t GetMaxModelSplitNum() const;

  virtual int64_t GetAreaId() const = 0;
  virtual bool MayConsumeModelDiff() const { return false; }

 protected:
  LogicalNode() : main_model_parallel_(nullptr) {}
  virtual CompTaskNode* NewCompTaskNode() const = 0;
  virtual void FixCompTaskNode(CompTaskNode*) const {}

 private:
  bool HasOpWithCondition(std::function<bool(const Operator*)>) const;

  std::vector<std::shared_ptr<Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
  std::shared_ptr<const std::vector<LogicalNode*>> shared_model_nodes_;
  LogicalNode* main_model_parallel_;

  HashMap<const LogicalNode*, std::vector<LogicalBlobId>> dst2data_lbis_;
};

#define BLD_SUB_TSK_GPH_MTHD_ARGS()                                                       \
  (const LogicalNode* src_logical, const LogicalNode* dst_logical,                        \
   const std::vector<CompTaskNode*>& sorted_src_comp_tasks,                               \
   const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,                               \
   HashMap<const LogicalNode*, std::vector<TaskNode*>>* logical2sorted_in_box,            \
   HashMap<const LogicalNode*, std::vector<TaskNode*>>* logical2sorted_out_box,           \
   std::function<TaskNode**(CompTaskNode * src, int64_t machine_id, int32_t mem_zone_id)> \
       MutBufTask,                                                                        \
   std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly)

class TaskGraph;
using BldSubTskGphMthd = void(TaskGraph::*) BLD_SUB_TSK_GPH_MTHD_ARGS();

class LogicalEdge final : public Edge<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalEdge);
  LogicalEdge() = default;
  ~LogicalEdge() = default;

  const LogicalBlobId& SoleLbi() const {
    CHECK_EQ(lbis_.size(), 1);
    return lbis_.front();
  }

  const std::vector<LogicalBlobId>& lbis() const { return lbis_; }
  std::vector<LogicalBlobId>& mut_lbis() { return lbis_; }

 private:
  std::vector<LogicalBlobId> lbis_;
};

BldSubTskGphMthd GetMthdForBldSubTskGph(const LogicalNode* src, const LogicalNode* dst);

using BldBoxingOpConfMthd = void (BoxingTaskNode::*)(
    const LogicalBlobId& lbi, const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
    const LogicalNode* in_logical, const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
    const LogicalNode* out_logical, BoxingOpConf*);
BldBoxingOpConfMthd GetMthdForBldBoxingOpConf(const LogicalNode* src, const LogicalNode* dst);

#define OVERRIDE_PURE_VIRTUAL_METHOD()            \
  std::string TypeName() const override;          \
  CompTaskNode* NewCompTaskNode() const override; \
  int64_t GetAreaId() const override;

#define LOGICAL_NODE_BOILERPLATE(class_name) \
  OF_DISALLOW_COPY_AND_MOVE(class_name);     \
  class_name() = default;                    \
  ~class_name() = default;                   \
  OVERRIDE_PURE_VIRTUAL_METHOD();

class BackwardLogicalNode;

class ForwardLogicalNode : public LogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardLogicalNode);
  ForwardLogicalNode() : bw_node_(nullptr) {}
  virtual ~ForwardLogicalNode() = default;

  BackwardLogicalNode* bw_node() const { return bw_node_; }

  BackwardLogicalNode* NewBackwardNode();

 protected:
  virtual BackwardLogicalNode* NewCorrectBackwardNode() = 0;

 private:
  BackwardLogicalNode* bw_node_;
};

class NormalForwardLogicalNode final : public ForwardLogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(NormalForwardLogicalNode);

  BackwardLogicalNode* NewCorrectBackwardNode() override;

 private:
};

int64_t NewAreaId();

#define LOGICAL_NODE_WITH_NEW_AREA_ID_BOILERPLATE(name)                             \
 public:                                                                            \
  OF_DISALLOW_COPY_AND_MOVE(name##LogicalNode);                                     \
  name##LogicalNode() { area_id_ = NewAreaId(); }                                   \
  ~name##LogicalNode() = default;                                                   \
                                                                                    \
  std::string TypeName() const override { return #name; }                           \
  CompTaskNode* NewCompTaskNode() const override { return new name##CompTaskNode; } \
  int64_t GetAreaId() const override { return area_id_; }                           \
                                                                                    \
 private:                                                                           \
  int64_t area_id_;

#define DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(name) \
  class name##LogicalNode final : public ForwardLogicalNode {       \
    LOGICAL_NODE_WITH_NEW_AREA_ID_BOILERPLATE(name)                 \
                                                                    \
   private:                                                         \
    BackwardLogicalNode* NewCorrectBackwardNode() override;         \
  }

DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(UnpackForward);

class PackForwardLogicalNode final : public ForwardLogicalNode {
  LOGICAL_NODE_WITH_NEW_AREA_ID_BOILERPLATE(PackForward)

 public:
  const UnpackForwardLogicalNode* related_unpack() const { return related_unpack_; }
  void set_related_unpack(UnpackForwardLogicalNode* val) { related_unpack_ = val; }

 private:
  BackwardLogicalNode* NewCorrectBackwardNode() override;

  UnpackForwardLogicalNode* related_unpack_;
};

class BackwardLogicalNode : public LogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardLogicalNode);
  BackwardLogicalNode() : fw_node_(nullptr) {}
  virtual ~BackwardLogicalNode() = default;

  ForwardLogicalNode* fw_node() const { return fw_node_; }

 private:
  friend class ForwardLogicalNode;

  ForwardLogicalNode* fw_node_;
};

class NormalBackwardLogicalNode final : public BackwardLogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(NormalBackwardLogicalNode);
};

#define DECLARE_DERIVED_BACKWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(name) \
  class name##LogicalNode final : public BackwardLogicalNode {       \
    LOGICAL_NODE_WITH_NEW_AREA_ID_BOILERPLATE(name);                 \
  }

DECLARE_DERIVED_BACKWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(UnpackBackward);

#define DECLARE_NAIVE_LOGICAL_NODE(name)  \
  class name final : public LogicalNode { \
   public:                                \
    LOGICAL_NODE_BOILERPLATE(name);       \
  }

DECLARE_NAIVE_LOGICAL_NODE(RecordLoadLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(DecodeLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(DecodeRandomLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(PrintLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(LossLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(LossAccLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(LossPrintLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(AccuracyLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(AccuracyAccLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(AccuracyPrintLogicalNode);

class NormalMdUpdtLogicalNode final : public LogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdtLogicalNode);
  NormalMdUpdtLogicalNode() : random_seed_(NewRandomSeed()) {}
  ~NormalMdUpdtLogicalNode() = default;

  OVERRIDE_PURE_VIRTUAL_METHOD();
  bool MayConsumeModelDiff() const override { return true; }

 private:
  void FixCompTaskNode(CompTaskNode*) const override;

  uint32_t random_seed_;
};

DECLARE_NAIVE_LOGICAL_NODE(MdSaveLogicalNode);

class MdDiffAccLogicalNode final : public LogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(MdDiffAccLogicalNode);
  bool MayConsumeModelDiff() const override { return true; }
};

class ReduceLogicalNode : public LogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceLogicalNode);
  ~ReduceLogicalNode() override = default;

  ReduceRankCtx& mut_rank_ctx() { return rank_ctx_; }
  const ReduceRankCtx& rank_ctx() const { return rank_ctx_; }

 protected:
  ReduceLogicalNode() = default;

 private:
  ReduceRankCtx rank_ctx_;
  void FixCompTaskNode(CompTaskNode* task_node) const override {
    task_node->mut_parallel_ctx()->mutable_rank_ctx()->set_rank_id(
        rank_ctx().Rank4ParallelId(task_node->parallel_id()));
    task_node->mut_parallel_ctx()->mutable_rank_ctx()->set_rank_num(rank_ctx().StageSegmentCount());
    int64_t rank_set_id =
        ((node_id() << 32) | rank_ctx().RankSet4ParallelId(task_node->parallel_id()));
    task_node->mut_parallel_ctx()->mutable_rank_ctx()->set_rank_set_id(rank_set_id);
  }
};

#define DECLARE_REDUCE_LOGICAL_NODE(name, may_consume_md_diff)                \
  class name final : public ReduceLogicalNode {                               \
   public:                                                                    \
    LOGICAL_NODE_BOILERPLATE(name);                                           \
    bool MayConsumeModelDiff() const override { return may_consume_md_diff; } \
  }

DECLARE_REDUCE_LOGICAL_NODE(ReduceConcatLogicalNode, true);
DECLARE_REDUCE_LOGICAL_NODE(ReduceSplitLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(ReduceScatterLogicalNode, true);
DECLARE_REDUCE_LOGICAL_NODE(ReduceGatherLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(NcclAllReduceLogicalNode, true);
DECLARE_REDUCE_LOGICAL_NODE(ReduceAddLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(NcclAllGatherLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(NcclReduceScatterLogicalNode, true);

DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(RepeatForward);
DECLARE_DERIVED_BACKWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(RepeatBackward);

class ReduceIdentityLogicalNode final : public LogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(ReduceIdentityLogicalNode);

  void set_fw_logical_nodes(const std::vector<LogicalNode*>& fw_logical_nodes) {
    fw_logical_nodes_ = fw_logical_nodes;
  }
  const LogicalNode* first_fw_logical_node() const { return fw_logical_nodes_.at(0); }
  bool MayConsumeModelDiff() const override { return true; }

 private:
  std::vector<LogicalNode*> fw_logical_nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_
