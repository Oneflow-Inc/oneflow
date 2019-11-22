#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_

#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/reduce_rank_context.h"
#include "oneflow/core/graph/pack_forward_task_node.h"
#include "oneflow/core/graph/unpack_forward_task_node.h"
#include "oneflow/core/graph/wait_and_send_ids_compute_task_node.h"
#include "oneflow/core/graph/foreign_input_compute_task_node.h"
#include "oneflow/core/graph/foreign_output_compute_task_node.h"
#include "oneflow/core/graph/callback_notify_compute_task_node.h"
#include "oneflow/core/graph/reentrant_lock_compute_task_node.h"
#include "oneflow/core/graph/source_tick_compute_task_node.h"
#include "oneflow/core/graph/tick_compute_task_node.h"
#include "oneflow/core/graph/acc_tick_compute_task_node.h"
#include "oneflow/core/graph/repeat_forward_compute_task_node.h"
#include "oneflow/core/graph/acc_compute_task_node.h"
#include "oneflow/core/graph/every_nth_compute_task_node.h"
#include "oneflow/core/graph/case_compute_task_node.h"
#include "oneflow/core/graph/esac_compute_task_node.h"

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

  // time_shape
  const Shape* out_blob_time_shape() const { return out_blob_time_shape_.get(); }
  void reset_out_blob_time_shape(const Shape* time_shape) {
    out_blob_time_shape_.reset(time_shape);
  }
  const Shape* in_blob_fastest_time_shape() const { return in_blob_fastest_time_shape_.get(); }
  void reset_in_blob_fastest_time_shape(const Shape* time_shape) {
    in_blob_fastest_time_shape_.reset(time_shape);
  }

  // Lbis
  size_t produced_batch_dim_lbis_cnt() const { return produced_batch_dim_lbis_cnt_; }
  size_t consumed_batch_dim_lbis_cnt() const { return consumed_batch_dim_lbis_cnt_; }
  std::vector<LogicalBlobId> GetLbisTo(const LogicalNode* dst) const;
  void SetDataLbisTo(const LogicalNode* dst, const std::vector<LogicalBlobId>&);
  bool IsDataLbiOnOutEdge(const LogicalBlobId& lbi) const;
  void set_produced_batch_dim_lbis_cnt(size_t val) { produced_batch_dim_lbis_cnt_ = val; }
  void set_consumed_batch_dim_lbis_cnt(size_t val) { consumed_batch_dim_lbis_cnt_ = val; }

  // util
  virtual std::string TypeName() const = 0;
  std::string VisualStr() const;
  void GenSortedCompTaskNodes(std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly,
                              std::vector<std::pair<int64_t, CompTaskNode*>>* nodes,
                              std::function<void(CompTaskNode*)>) const;

  // other
  virtual int64_t GetAreaId() const = 0;
  virtual bool MayConsumeModelDiff() const { return false; }

 protected:
  LogicalNode() {}
  virtual CompTaskNode* NewCompTaskNode() const = 0;
  virtual void FixCompTaskNode(CompTaskNode*) const {}

 private:
  bool HasOpWithCondition(std::function<bool(const Operator*)>) const;

  std::vector<std::shared_ptr<Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;

  HashMap<const LogicalNode*, std::vector<LogicalBlobId>> dst2data_lbis_;
  std::unique_ptr<const Shape> in_blob_fastest_time_shape_;
  std::unique_ptr<const Shape> out_blob_time_shape_;
  size_t produced_batch_dim_lbis_cnt_;
  size_t consumed_batch_dim_lbis_cnt_;
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

class ForwardLogicalNode : public LogicalNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardLogicalNode);
  ForwardLogicalNode() = default;
  virtual ~ForwardLogicalNode() = default;
};

class NormalForwardLogicalNode final : public ForwardLogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(NormalForwardLogicalNode);

 private:
};

class OptimizerLogicalNode final : public ForwardLogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(OptimizerLogicalNode);

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
  }

DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(UnpackForward);

class PackForwardLogicalNode final : public ForwardLogicalNode {
  LOGICAL_NODE_WITH_NEW_AREA_ID_BOILERPLATE(PackForward)

 public:
  const UnpackForwardLogicalNode* related_unpack() const { return related_unpack_; }
  void set_related_unpack(UnpackForwardLogicalNode* val) { related_unpack_ = val; }

 private:
  UnpackForwardLogicalNode* related_unpack_;
};

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
DECLARE_NAIVE_LOGICAL_NODE(AccuracyLogicalNode);

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
DECLARE_REDUCE_LOGICAL_NODE(ReduceScatterLogicalNode, true);
DECLARE_REDUCE_LOGICAL_NODE(ReduceGatherLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(NcclAllReduceLogicalNode, true);
DECLARE_REDUCE_LOGICAL_NODE(ReduceAddLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(NcclAllGatherLogicalNode, false);
DECLARE_REDUCE_LOGICAL_NODE(NcclReduceScatterLogicalNode, true);

DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(WaitAndSendIds);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(ForeignInput);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(ForeignOutput);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(CallbackNotify);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(ReentrantLock);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(SourceTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(AccTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Tick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(RepeatForward);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Acc);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(EveryNth);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Case);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Esac);

#define DECLARE_BEFORE_OR_AFTER_ALLREDUCE_REDUCE_NODE(class_name, may_consume_md_diff) \
  class class_name final : public ReduceLogicalNode {                                  \
   public:                                                                             \
    LOGICAL_NODE_BOILERPLATE(class_name);                                              \
    void set_order_in_logical_graph(int32_t order_in_logical_graph) {                  \
      order_in_logical_graph_ = order_in_logical_graph;                                \
    }                                                                                  \
    int32_t order_in_logical_graph() const;                                            \
    bool MayConsumeModelDiff() const override { return may_consume_md_diff; }          \
                                                                                       \
   private:                                                                            \
    int32_t order_in_logical_graph_;                                                   \
  }

DECLARE_BEFORE_OR_AFTER_ALLREDUCE_REDUCE_NODE(ReduceIdentityLogicalNode, true);
DECLARE_BEFORE_OR_AFTER_ALLREDUCE_REDUCE_NODE(ReduceSplitLogicalNode, false);

#define DECLARE_FACADE_LOGICAL_NODE(class_name)                         \
  class class_name final : public LogicalNode {                         \
   public:                                                              \
    OF_DISALLOW_COPY_AND_MOVE(class_name);                              \
    class_name() = default;                                             \
    ~class_name() override = default;                                   \
    std::string TypeName() const override { return #class_name; }       \
    CompTaskNode* NewCompTaskNode() const override { UNIMPLEMENTED(); } \
    int64_t GetAreaId() const override { UNIMPLEMENTED(); };            \
  }

DECLARE_FACADE_LOGICAL_NODE(AllReduceFacadeLogicalNode);

class NcclTupleBroadcastLogicalNode : public LogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(NcclTupleBroadcastLogicalNode);

 private:
  void FixCompTaskNode(CompTaskNode* task_node) const override {
    RankContext* rank_ctx = task_node->mut_parallel_ctx()->mutable_rank_ctx();
    rank_ctx->set_rank_set_id(node_id() << 32);
    rank_ctx->set_rank_id(task_node->parallel_ctx()->parallel_id());
    rank_ctx->set_rank_num(task_node->parallel_ctx()->parallel_num());
  }
};

class NcclTupleReduceLogicalNode : public LogicalNode {
 public:
  LOGICAL_NODE_BOILERPLATE(NcclTupleReduceLogicalNode);

 private:
  void FixCompTaskNode(CompTaskNode* task_node) const override {
    RankContext* rank_ctx = task_node->mut_parallel_ctx()->mutable_rank_ctx();
    rank_ctx->set_rank_set_id(node_id() << 32);
    rank_ctx->set_rank_id(task_node->parallel_ctx()->parallel_id());
    rank_ctx->set_rank_num(task_node->parallel_ctx()->parallel_num());
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_
