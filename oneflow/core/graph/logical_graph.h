#ifndef ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class LogicalGraph final : public Graph<LogicalNode, LogicalEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalGraph);
  LogicalGraph() = delete;
  ~LogicalGraph() = default;

  LogicalGraph(bool is_train);

  const char* TypeName() const override { return "LogicalGraph"; }
  int64_t total_mbn_num() const { return total_mbn_num_; }

 private:
  struct BackwardCloneInfo {
    LogicalNode* succ_node;
    LogicalBlobId lbi;
    std::vector<LogicalEdge*> edges;
  };
  struct ReduceCtx {
    std::vector<LogicalNode*> fw_logicals;
    std::vector<LogicalNode*> bw_logicals;
    std::vector<LogicalNode*> md_diff_acc_logicals;
    std::vector<LogicalNode*> md_updt_logicals;
  };
  template<typename LogicalNodeType>
  void ForEachLogicalNode(std::function<void(LogicalNodeType*)> Handler);
  void GroupNodesForReduceStruct();

  void BuildFwStruct();
  void NaiveBuildFwStruct(HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes);
  void FixSharedModelNodes(const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes);
  void ReConnectToFwClone(LogicalNode* clone_node, const LogicalBlobId& lbi,
                          const std::vector<LogicalEdge*>& edges, const std::string& obn);
  void SetMainModelParallel();
  void BuildBwStruct();
  void NaiveBuildBwStruct();
  void AddBackwardClone();
  void AddOneBackwardClone(const BackwardCloneInfo& clone_info);

  void MergeEdge();
  void SetNodeDataLbi();
  void BuildLossPrintStruct();
  void BuildAccuracyPrintStruct();
  void BuildModelStruct(bool is_train);
  void AddReduceSwitch(LogicalNode* src, LogicalNode* dst);
  void AddReduceScatterAddGatherNodes2(LogicalNode* src, LogicalNode* dst);
  void BuildReduceStruct(const ReduceCtx& reduce_ctx);
  void SetupNormalMdUpdtOp();
  MdSaveLogicalNode* BuildMdSaveStruct(const ForwardLogicalNode* fw_logical,
                                       LogicalNode* need_save_logical);
  NormalMdUpdtLogicalNode* BuildNormalMdUpdtAndMdSaveStruct(bool is_train,
                                                            ForwardLogicalNode* fw_logical);
  void ConnectFwToBw();
  void UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn);
  void UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn);

  int64_t total_mbn_num_;

  std::vector<std::vector<const LogicalNode*>> fw_node_groups_;
  HashMap<const LogicalEdge*, std::string> edge2ibn_;
  HashMap<const LogicalEdge*, std::string> edge2obn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
