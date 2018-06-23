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

  void BuildFwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn);
  void NaiveBuildFwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn,
                          HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes);
  void FixSharedModelNodes(const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes);
  void AddB121Clone(HashMap<LogicalEdge*, std::string>* edge2ibn);
  void CollectB121CloneInfos(std::vector<B121CloneInfo>* clone_infos);
  void AddOneB121CloneNode(const B121CloneInfo& clone_info,
                           HashMap<LogicalEdge*, std::string>* edge2ibn);
  void ReConnectToFwClone(LogicalNode* clone_node, const LogicalBlobId& lbi,
                          const std::vector<LogicalEdge*>& edges,
                          const HashMap<LogicalEdge*, std::string>& edge2ibn);
  void SetMainModelParallel();
  void BuildBwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn);
  void NaiveBuildBwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn);
  void AddBackwardClone(const HashMap<LogicalEdge*, std::string>& edge2ibn);
  void AddOneBackwardClone(const BackwardCloneInfo& clone_info,
                           const HashMap<LogicalEdge*, std::string>& edge2ibn);
  void MergeEdge();
  void SetNodeDataLbi();
  void BuildLossPrintStruct();
  void BuildModelStruct(bool is_train);
  void BuildReduceStruct(LogicalNode* src, LogicalNode* dst);
  void SetupNormalMdUpdtOp();
  MdSaveLogicalNode* BuildMdSaveStruct(const ForwardLogicalNode* fw_logical,
                                       LogicalNode* need_save_logical);
  NormalMdUpdtLogicalNode* BuildNormalMdUpdtAndMdSaveStruct(bool is_train,
                                                            ForwardLogicalNode* fw_logical);
  void BuildRecordLoadStruct();
  void ConnectFwToBw();

  int64_t total_mbn_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOGICAL_GRAPH_H_
