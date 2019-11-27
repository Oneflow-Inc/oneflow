#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class LogicalGraph final : public Graph<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalGraph);
  LogicalGraph() = delete;
  ~LogicalGraph() = default;

  LogicalGraph(const Job& job);

  const char* TypeName() const override { return "LogicalGraph"; }

  void ForEachNecessaryCtrlEdge(
      const std::function<void(const LogicalNode* src, const LogicalNode* dst,
                               int64_t ctrl_regst_num)>& Handler) const;

 private:
  template<typename LogicalNodeType>
  void ForEachLogicalNode(std::function<void(LogicalNodeType*)> Handler);

  void BuildFwStruct();
  void NaiveBuildFwStruct(HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes);
  void LinkUnpackFw2PackFw(const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes);

  void MergeEdge();
  void SetNodeDataLbi();
  void AddReduceScatterAddGatherNodes(LogicalNode* src, LogicalNode* dst,
                                      const ReduceRankCtx& prev_rank_ctx);
  void AddAllReduce(LogicalNode* src, LogicalNode* dst);
  void AddNcclAllReduce(LogicalNode* src, LogicalNode* dst);
  void AddNcclReduceScatterAndAllGather(LogicalNode* src, LogicalNode* dst);
  void ReplaceAllReduceFacades();

  void UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn);
  void UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn);

  bool MustHaveModelDiffAcc();

  Job job_;

  HashMap<const LogicalEdge*, std::string> edge2ibn_;
  HashMap<const LogicalEdge*, std::string> edge2obn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
