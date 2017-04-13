#ifndef ONEFLOW_GRAPH_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_COMP_TASK_NODE_H_

#include <algorithm>
#include "graph/task_node.h"

namespace oneflow {

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  int32_t parallel_id() const { return parallel_id_; }
  void set_parallel_id(int32_t parallel_id) { parallel_id_ = parallel_id; }

  bool IsLossNode() const { TODO(); }

  bool IsFaker() const { return chain_node()->IsFaker(); }

  void DataFwBuildExecAndProducedRegsts(Path*);
  void ModelUpdateFwBuildExecAndProducedRegsts(Path*);
  void ModelLoadFwBuildExecAndProducedRegsts(Path*);
  void ModelSaveFwBuildExecAndProducedRegsts(Path*);

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }
  virtual CopyOpConf::CopyType CopyInOpType() = 0;

 private:
  using Lbn2NodeObnMap =
      HashMap<std::string, std::pair<ExecNode*, std::string>>;
  using Lbn2NodeIbnVecMap =
      HashMap<std::string, std::vector<std::pair<ExecNode*, std::string>>>;
  struct CloneInfo {
    std::string lbn;
    std::shared_ptr<const Operator> clone_op;
    ExecNode* pred_node;
    std::vector<ExecEdge*> edges;
  };

  void FwBuildExecAndProducedRegsts(Path*) override;
  void FwBuildFromUserOps(
      Lbn2NodeObnMap* lbn2producer,
      Lbn2NodeIbnVecMap* extern_in_lbn2consumers);
  void FwAddCopyInOp(Lbn2NodeIbnVecMap* extern_in_lbn2consumers);
  void FwAddCloneOp();
  void FwCollectCloneInfoVec(std::vector<CloneInfo>* clone_info_vec);
  void FwAddOneCloneNode(const CloneInfo& clone_info);
  void FwBindOutEdgeAndRegst();
  void FwSetProducedRegstDescs(
      const Lbn2NodeObnMap& lbn2producer,
      const Lbn2NodeIbnVecMap& extern_in_lbn2consumers);
  void BpBuildExecAndProducedRegsts(Path*) override;
  void BpBuildExecGraph(
      const ExecGraph& fw_gph,
      const ExecNode* cp_in_node,
      HashMap<const ExecNode*, ExecNode*>* fw_node2bp_node);
  void BpSetProducedRegstDescs(
      const ExecNode* cp_in_node,
      const HashMap<const ExecNode*, ExecNode*>& fw_node2bp_node);

  int32_t parallel_id_;

};

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec);

class HostCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompTaskNode);
  HostCompTaskNode() = default;
  ~HostCompTaskNode() = default;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new HostCompTaskNode);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::H2H;
  }

};

class DeviceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTaskNode);
  DeviceCompTaskNode() = default;
  ~DeviceCompTaskNode() = default;
  
 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new DeviceCompTaskNode);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::D2D;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMP_TASK_NODE_H_
