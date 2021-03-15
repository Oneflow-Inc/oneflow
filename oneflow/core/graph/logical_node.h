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
  const Shape* out_blob_time_shape() const { return CHECK_JUST(SoleOp()->GetOpTimeShape()).get(); }
  const Shape* in_blob_fastest_time_shape() const {
    return CHECK_JUST(SoleOp()->GetInputBlobFastestTimeShape()).get();
  }

  // Lbis
  std::vector<LogicalBlobId> GetLbisTo(const LogicalNode* dst) const;
  void SetDataLbisTo(const LogicalNode* dst, const std::vector<LogicalBlobId>&);
  bool IsDataLbiOnOutEdge(const LogicalBlobId& lbi) const;

  // util
  virtual std::string TypeName() const = 0;
  std::string VisualStr() const;
  void GenSortedCompTaskNodes(std::function<void(CompTaskNode*)>) const;

 protected:
  LogicalNode() {}
  virtual CompTaskNode* NewCompTaskNode() const = 0;

 private:
  bool HasOpWithCondition(std::function<bool(const Operator*)>) const;

  std::vector<std::shared_ptr<const Operator>> op_vec_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;

  HashMap<const LogicalNode*, std::vector<LogicalBlobId>> dst2data_lbis_;
};

#define BLD_SUB_TSK_GPH_MTHD_ARGS()                                                       \
  (const LogicalNode* src_logical, const LogicalNode* dst_logical,                        \
   const std::vector<CompTaskNode*>& sorted_src_comp_tasks,                               \
   const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,                               \
   std::function<TaskNode**(CompTaskNode * src, int64_t machine_id, int32_t mem_zone_id)> \
       MutBufTask)

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

#define DECLARE_LOGICAL_NODE(name)                     \
  class name##LogicalNode final : public LogicalNode { \
   public:                                             \
    OF_DISALLOW_COPY_AND_MOVE(name##LogicalNode);      \
    name##LogicalNode() = default;                     \
    ~name##LogicalNode() = default;                    \
    std::string TypeName() const override;             \
    CompTaskNode* NewCompTaskNode() const override;    \
  };

DECLARE_LOGICAL_NODE(NormalForward);

#define LOGICAL_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(WaitAndSendIds)   \
  OF_PP_MAKE_TUPLE_SEQ(ForeignInput)     \
  OF_PP_MAKE_TUPLE_SEQ(ForeignOutput)    \
  OF_PP_MAKE_TUPLE_SEQ(CallbackNotify)   \
  OF_PP_MAKE_TUPLE_SEQ(ReentrantLock)    \
  OF_PP_MAKE_TUPLE_SEQ(SrcSubsetTick)    \
  OF_PP_MAKE_TUPLE_SEQ(DstSubsetTick)    \
  OF_PP_MAKE_TUPLE_SEQ(SourceTick)       \
  OF_PP_MAKE_TUPLE_SEQ(AccTick)          \
  OF_PP_MAKE_TUPLE_SEQ(Tick)             \
  OF_PP_MAKE_TUPLE_SEQ(DeviceTick)       \
  OF_PP_MAKE_TUPLE_SEQ(Case)             \
  OF_PP_MAKE_TUPLE_SEQ(Esac)             \
  OF_PP_MAKE_TUPLE_SEQ(DecodeH2D)        \
  OF_PP_MAKE_TUPLE_SEQ(DistributeConcat) \
  OF_PP_MAKE_TUPLE_SEQ(DistributeSplit)  \
  OF_PP_MAKE_TUPLE_SEQ(DecodeRandom)

OF_PP_FOR_EACH_TUPLE(DECLARE_LOGICAL_NODE, LOGICAL_TYPE_SEQ);

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
