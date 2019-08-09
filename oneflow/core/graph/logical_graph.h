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
  int64_t total_mbn_num() const { return total_mbn_num_; }

  void ForEachNecessaryCtrlEdge(
      const std::function<void(const LogicalNode* src, const LogicalNode* dst,
                               int64_t ctrl_regst_num)>& Handler) const;

 private:
  struct BackwardCloneInfo {
    LogicalNode* succ_node;
    LogicalBlobId lbi;
    std::vector<LogicalEdge*> edges;
  };
  struct ReduceCtx {
    int32_t order_in_logical_graph;
    std::vector<LogicalNode*> fw_logicals;
    std::vector<LogicalNode*> bw_logicals;
    std::vector<LogicalNode*> md_diff_acc_logicals;
    std::vector<LogicalNode*> md_updt_logicals;
  };
  template<typename LogicalNodeType>
  void ForEachLogicalNode(std::function<void(LogicalNodeType*)> Handler);

  void BuildFwStruct();
  void NaiveBuildFwStruct(HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes);
  void LinkUnpackFw2PackFw(const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes);

  void MergeEdge();
  void SetNodeDataLbi();
  void BuildModelStruct(bool is_train);
  void AddReduceScatterAddGatherNodes(LogicalNode* src, LogicalNode* dst,
                                      const ReduceRankCtx& prev_rank_ctx);
  void AddAllReduce(LogicalNode* src, LogicalNode* dst);
  void AddNcclAllReduce(LogicalNode* src, LogicalNode* dst);
  void AddCudaRingAllReduce(LogicalNode* src, LogicalNode* dst);
  void AddNcclReduceScatterAndAllGather(LogicalNode* src, LogicalNode* dst);
  void SetupNormalMdUpdtOp();
  void ReplaceAllReduceFacades();

  void UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn);
  void UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn);

  bool MustHaveModelDiffAcc();

  Job job_;
  int64_t total_mbn_num_;

  std::vector<std::vector<const LogicalNode*>> fw_node_groups_;
  HashMap<const LogicalEdge*, std::string> edge2ibn_;
  HashMap<const LogicalEdge*, std::string> edge2obn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
