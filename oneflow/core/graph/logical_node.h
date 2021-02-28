/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/wait_and_send_ids_compute_task_node.h"
#include "oneflow/core/graph/foreign_input_compute_task_node.h"
#include "oneflow/core/graph/foreign_output_compute_task_node.h"
#include "oneflow/core/graph/callback_notify_compute_task_node.h"
#include "oneflow/core/graph/reentrant_lock_compute_task_node.h"
#include "oneflow/core/graph/src_subset_tick_compute_task_node.h"
#include "oneflow/core/graph/dst_subset_tick_compute_task_node.h"
#include "oneflow/core/graph/source_tick_compute_task_node.h"
#include "oneflow/core/graph/tick_compute_task_node.h"
#include "oneflow/core/graph/device_tick_compute_task_node.h"
#include "oneflow/core/graph/acc_tick_compute_task_node.h"
#include "oneflow/core/graph/case_compute_task_node.h"
#include "oneflow/core/graph/esac_compute_task_node.h"
#include "oneflow/core/graph/decode_h2d_compute_task_node.h"

namespace oneflow {

class LogicalEdge;

class LogicalNode : public Node<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalNode);
  virtual ~LogicalNode() = default;

  // op_vec_
  std::shared_ptr<const Operator> SoleOp() const;
  const std::vector<std::shared_ptr<const Operator>>& op_vec() const { return op_vec_; }
  std::vector<std::shared_ptr<const Operator>>& mut_op_vec() { return op_vec_; }

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
  std::vector<LogicalBlobId> GetLbisTo(const LogicalNode* dst) const;
  void SetDataLbisTo(const LogicalNode* dst, const std::vector<LogicalBlobId>&);
  bool IsDataLbiOnOutEdge(const LogicalBlobId& lbi) const;

  // util
  virtual std::string TypeName() const = 0;
  std::string VisualStr() const;
  void GenSortedCompTaskNodes(std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly,
                              std::vector<std::pair<int64_t, CompTaskNode*>>* nodes,
                              std::function<void(CompTaskNode*)>) const;

  // other
  virtual int64_t GetAreaId() const = 0;

 protected:
  LogicalNode() {}
  virtual CompTaskNode* NewCompTaskNode() const = 0;

 private:
  bool HasOpWithCondition(std::function<bool(const Operator*)>) const;

  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;

  HashMap<const LogicalNode*, std::vector<LogicalBlobId>> dst2data_lbis_;
  std::unique_ptr<const Shape> in_blob_fastest_time_shape_;
  std::unique_ptr<const Shape> out_blob_time_shape_;
};

#define BLD_SUB_TSK_GPH_MTHD_ARGS()                                                       \
  (const LogicalNode* src_logical, const LogicalNode* dst_logical,                        \
   const std::vector<CompTaskNode*>& sorted_src_comp_tasks,                               \
   const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,                               \
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

#define DECLARE_NAIVE_LOGICAL_NODE(name)  \
  class name final : public LogicalNode { \
   public:                                \
    LOGICAL_NODE_BOILERPLATE(name);       \
  }

DECLARE_NAIVE_LOGICAL_NODE(DecodeRandomLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(DistributeConcatLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(DistributeSplitLogicalNode);
DECLARE_NAIVE_LOGICAL_NODE(PrintLogicalNode);

DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(WaitAndSendIds);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(ForeignInput);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(ForeignOutput);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(CallbackNotify);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(ReentrantLock);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(SrcSubsetTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(DstSubsetTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(SourceTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(AccTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Tick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(DeviceTick);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Case);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(Esac);
DECLARE_DERIVED_FORWARD_LOGICAL_NODE_WITH_NEW_AREA_ID(DecodeH2D);

class UserOpAreaIdCreator {
 public:
  virtual ~UserOpAreaIdCreator() = default;
  virtual int64_t GetAreaId() = 0;
};

class FixedUserOpAreaIdCreator : public UserOpAreaIdCreator {
 public:
  explicit FixedUserOpAreaIdCreator(int64_t area_id) : area_id_(area_id) {}
  ~FixedUserOpAreaIdCreator() override = default;

  int64_t GetAreaId() override { return area_id_; }

 private:
  int64_t area_id_;
};

class IndependentUserOpAreaIdCreator : public UserOpAreaIdCreator {
 public:
  IndependentUserOpAreaIdCreator() = default;
  ~IndependentUserOpAreaIdCreator() override = default;

  int64_t GetAreaId() override { return NewAreaId(); }
};

#define REGISTER_USER_OP_AREA_ID(op_type_name, area_id)                  \
  REGISTER_CLASS_CREATOR(std::string, op_type_name, UserOpAreaIdCreator, \
                         ([] { return new FixedUserOpAreaIdCreator(area_id); }));

#define REGISTER_USER_OP_INDEPENDENT_AREA_ID(op_type_name)               \
  REGISTER_CLASS_CREATOR(std::string, op_type_name, UserOpAreaIdCreator, \
                         ([] { return new IndependentUserOpAreaIdCreator(); }));

class UserOpCompTaskNodeCreator {
 public:
  virtual ~UserOpCompTaskNodeCreator() = default;
  virtual CompTaskNode* NewCompTaskNode(const OperatorConf& op_conf) = 0;
};

template<typename CompTaskNodeType>
class StaticUserOpCompTaskNodeCreator : public UserOpCompTaskNodeCreator {
 public:
  StaticUserOpCompTaskNodeCreator() = default;
  ~StaticUserOpCompTaskNodeCreator() override = default;

 private:
  CompTaskNode* NewCompTaskNode(const OperatorConf& op_conf) override {
    return new CompTaskNodeType();
  }
};

class FnUserOpCompTaskNodeCreator : public UserOpCompTaskNodeCreator {
 public:
  using CreateFn = std::function<CompTaskNode*(const OperatorConf& op_conf)>;
  explicit FnUserOpCompTaskNodeCreator(CreateFn fn) : fn_(std::move(fn)) {}
  ~FnUserOpCompTaskNodeCreator() override = default;

 private:
  CompTaskNode* NewCompTaskNode(const OperatorConf& op_conf) override { return fn_(op_conf); }
  CreateFn fn_;
};

#define REGISTER_USER_OP_COMP_TASK_NODE_TYPE(op_type_name, comp_task_node_type)               \
  REGISTER_CLASS_CREATOR(std::string, op_type_name, UserOpCompTaskNodeCreator, ([] {          \
                           return new StaticUserOpCompTaskNodeCreator<comp_task_node_type>(); \
                         }));
}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_NODE_H_
