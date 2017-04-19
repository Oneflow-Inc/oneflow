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

  int64_t parallel_id() const { return parallel_id_; }
  void set_parallel_id(int64_t parallel_id) { parallel_id_ = parallel_id; }

  bool IsLossNode() const { TODO(); }

  bool IsFaker() const { return chain_node()->IsFaker(); }

  void DataFwBuildExecAndProducedRegsts(TaskGraph*);
  void MdUpdtFwBuildExecAndProducedRegsts(TaskGraph*);
  void MdLoadFwBuildExecAndProducedRegsts(TaskGraph*);
  void MdSaveFwBuildExecAndProducedRegsts(TaskGraph*);

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }
  virtual CopyOpConf::CopyType CopyInOpType() = 0;

 private:
  using Lbn2NodeBnMap =
      HashMap<std::string, std::pair<ExecNode*, std::string>>;

  void FwBuildExecAndProducedRegsts(TaskGraph*) override;
  void FwBuildFromUserOps(
      Lbn2NodeBnMap* lbn2producer,
      Lbn2NodeBnMap* extern_in_lbn2consumer);
  void FwAddCopyInOp(Lbn2NodeBnMap* extern_in_lbn2consumer);
  void FwSetDataRegstDesc(
      const Lbn2NodeBnMap& lbn2producer,
      const Lbn2NodeBnMap& extern_in_lbn2consumer);
  void FwSetModelTmpRegstDesc();

  void BpBuildExecAndProducedRegsts(TaskGraph*) override;
  void BpBuildExecGraph(
      const ExecGraph& fw_gph,
      const ExecNode* cp_in_node,
      HashMap<const ExecNode*, ExecNode*>* fw_node2bp_node,
      HashMap<ExecEdge*, const ExecEdge*>* bp_edge2fw_edge);
  void BpSetDataDiffRegst(
      const ExecNode* cp_in_node,
      const HashMap<const ExecNode*, ExecNode*>& fw_node2bp_node,
      const HashMap<ExecEdge*, const ExecEdge*>& bp_edge2fw_edge);
  void BpSetModelDiffRegst();

  int64_t parallel_id_;

};

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec);

class HostCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompTaskNode);
  HostCompTaskNode() = default;
  ~HostCompTaskNode() = default;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return make_unique<HostCompTaskNode> ();
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
    return make_unique<DeviceCompTaskNode> ();
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
