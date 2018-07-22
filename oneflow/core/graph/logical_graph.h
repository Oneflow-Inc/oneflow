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
  struct B121CloneInfo {
    LogicalNode* pred_node;
    LogicalBlobId lbi;
    std::vector<LogicalEdge*> edges_boxing;
    std::vector<LogicalEdge*> edges_121;
  };
  struct BackwardCloneInfo {
    LogicalNode* succ_node;
    LogicalBlobId lbi;
    std::vector<LogicalEdge*> edges;
  };
  template<typename LogicalNodeType>
  void ForEachLogicalNode(std::function<void(LogicalNodeType*)> Handler);

  void BuildFwStruct();
  void NaiveBuildFwStruct(HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes);
  void FixSharedModelNodes(const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes);
  void AddB121Clone();
  void CollectB121CloneInfos(std::vector<B121CloneInfo>* clone_infos);
  void AddOneB121CloneNode(const B121CloneInfo& clone_info);
  void ReConnectToFwClone(LogicalNode* clone_node, const LogicalBlobId& lbi,
                          const std::vector<LogicalEdge*>& edges, const std::string& obn);
  void SetMainModelParallel();
  void BuildBwStruct();
  void NaiveBuildBwStruct();
  void AddBackwardClone();
  void AddOneBackwardClone(const BackwardCloneInfo& clone_info);
  void MoveBackwardActivations();
  void RemoveBackwardAdd();
  void CollectBackwardB121CloneInfos(HashMap<LogicalNode*, LogicalNode*>* bw_add_node2pre_node);
  void RemoveOneBackwardAdd(const std::pair<LogicalNode*, LogicalNode*>& bw_add_node_and_pre);

  void MergeEdge();
  void SetNodeDataLbi();
  void BuildLossPrintStruct();
  void BuildAccuracyPrintStruct();
  void BuildModelStruct(bool is_train);
  void BuildReduceStruct(LogicalNode* src, LogicalNode* dst);
  void SetupNormalMdUpdtOp();
  MdSaveLogicalNode* BuildMdSaveStruct(const ForwardLogicalNode* fw_logical,
                                       LogicalNode* need_save_logical);
  NormalMdUpdtLogicalNode* BuildNormalMdUpdtAndMdSaveStruct(bool is_train,
                                                            ForwardLogicalNode* fw_logical);
  void ConnectFwToBw();
  void UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn);
  void UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn);

  int64_t total_mbn_num_;

  HashMap<const LogicalEdge*, std::string> edge2ibn_;
  HashMap<const LogicalEdge*, std::string> edge2obn_;
  HashMap<LogicalNode*, LogicalNode*> bw_clone2fw_producer_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
