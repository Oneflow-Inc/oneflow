#ifndef ONEFLOW_GRAPH_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_COMP_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  bool HasOpWithOutDiff() const;
  bool HasOpWithIndiff() const;

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }
  virtual CopyOpConf::CopyType CopyInOpType() = 0;

 private:
  using Lbn2NodeMap = std::unordered_map<std::string, ExecNode*>;
  using Lbn2NodeVecMap = std::unordered_map<std::string, std::vector<ExecNode*>>;
  void FwBuildExecGraphAndSetProducedRegisterDescs() override;
  void FwBuildFromUserOps(
      Lbn2NodeMap* lbn2producer,
      Lbn2NodeVecMap* extern_in_lbn2consumers);
  void FwAddCopyInOp(Lbn2NodeVecMap* extern_in_lbn2consumers);
  void FwAddCloneOp();
  void FwSetOutEdgeRegisterPtr();
  void FwSetRegisterPtrs4ExecNodes(
      const Lbn2NodeMap& lbn2producer,
      const Lbn2NodeVecMap& extern_in_lbn2consumers);
  void FwSetProducedRegisterDescs();
  void BpBuildExecGraphAndSetProducedRegisterDescs() override;
  void BpBuildExecGraph(
      const ExecGraph& fw_graph,
      const ExecNode* cp_in_node,
      std::unordered_map<const ExecNode*, ExecNode*>* fw_node2bp_node);
  void BpSetOutEdgeRegisterPtr();
  void BpSetRegisterDescPtrs4Nodes(
      const ExecNode* cp_in_node,
      const std::unordered_map<const ExecNode*, ExecNode*>& fw_node2bp_node);
  void BpSetProducedRegisterDescs();

};

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
